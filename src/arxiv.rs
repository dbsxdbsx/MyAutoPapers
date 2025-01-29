use crate::types::Paper;
use crate::utils::remove_duplicated_spaces;
use anyhow::Result;
use chrono::{DateTime, Utc};
use feed_rs::parser;
use url::Url;

pub async fn request_paper_with_arxiv_api(
    keyword: &str,
    per_keyword_max_results: usize,
    link: &str,
) -> Result<Vec<Paper>> {
    let keyword = format!("\"{keyword}\"");
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=ti:{keyword}+{link}+abs:{keyword}&max_results={per_keyword_max_results}&sortBy=lastUpdatedDate"
    );

    println!("正在从 arXiv 获取论文数据...,url:{}", url);

    let url = Url::parse(&url)?;
    println!("正在从 arXiv 获取论文数据...");
    let response = reqwest::get(url).await?.text().await?;
    let feed = parser::parse(response.as_bytes())?;

    let papers: Vec<Paper> = feed
        .entries
        .into_iter()
        .map(|entry| {
            let authors = entry
                .authors
                .into_iter()
                .map(|author| remove_duplicated_spaces(&author.name))
                .collect();

            let tags = entry
                .categories
                .into_iter()
                .map(|category| remove_duplicated_spaces(&category.term))
                .collect();

            let title = match &entry.title {
                Some(t) => remove_duplicated_spaces(&t.content),
                None => String::new(),
            };

            let abstract_text = match &entry.summary {
                Some(s) => remove_duplicated_spaces(&s.content),
                None => String::new(),
            };

            let date = match &entry.published {
                Some(d) => d.to_rfc3339(),
                None => String::new(),
            };

            Paper {
                title,
                authors,
                abstract_text,
                link: remove_duplicated_spaces(&entry.links[0].href),
                tags,
                date,
            }
        })
        .collect();

    println!("成功获取 {} 篇论文", papers.len());
    Ok(papers)
}

pub async fn get_daily_papers_by_keyword_with_retries(
    keyword: &str,
    retry_times: usize,
    per_keyword_max_result: usize,
    link: &str,
) -> Result<Vec<Paper>> {
    let mut last_error = None;
    for i in 0..retry_times {
        match get_daily_papers_by_keyword(keyword, per_keyword_max_result, link).await {
            Ok(papers) if !papers.is_empty() => return Ok(papers),
            Ok(_) => {
                return Ok(vec![]);
            }
            Err(e) => {
                println!("第 {} 次尝试失败: {}，重试中...", i + 1, e);
                last_error = Some(e);
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            }
        }
    }
    Err(anyhow::anyhow!(
        "尝试 {} 次后仍未获取到论文{}",
        retry_times,
        last_error
            .map(|e| format!(", 最后一次错误: {e}"))
            .unwrap_or_default()
    ))
}

pub async fn get_daily_papers_by_keyword(
    keyword: &str,
    per_keyword_max_result: usize,
    link: &str,
) -> Result<Vec<Paper>> {
    println!("正在搜索组合关键词: {keyword}");

    // 拆分包含斜杠的关键词为多个子关键词
    let sub_keywords: Vec<&str> = keyword.split('/').collect();
    let mut all_papers = Vec::new();

    for sub_key in sub_keywords {
        println!("-> 处理子关键词: {sub_key}");
        let mut papers =
            request_paper_with_arxiv_api(sub_key, per_keyword_max_result, link).await?;
        all_papers.append(&mut papers);
    }

    // 去重处理（根据论文链接）
    println!("去重处理前论文数量: {}", all_papers.len());
    let mut seen = std::collections::HashSet::new();
    all_papers.retain(|paper| seen.insert(paper.link.clone()));
    println!("去重处理后论文数量: {}", all_papers.len());

    Ok(all_papers)
}

pub fn filter_papers(papers: Vec<Paper>, exclude_keywords: &[String]) -> Vec<Paper> {
    let mut excluded_count = 0;
    papers
        .into_iter()
        .filter(|paper| {
            let exclude = exclude_keywords.iter().any(|kw| {
                let found = paper.title.to_lowercase().contains(&kw.to_lowercase())
                    || paper
                        .abstract_text
                        .to_lowercase()
                        .contains(&kw.to_lowercase());
                if found {
                    excluded_count += 1;
                    println!(
                        "排除论文[{}]: {}\n原因: 包含关键词 '{}'",
                        excluded_count, paper.title, kw
                    );
                }
                found
            });
            !exclude
        })
        .collect()
}

impl Paper {
    pub fn date_time(&self) -> Option<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(&self.date)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
    }
}
