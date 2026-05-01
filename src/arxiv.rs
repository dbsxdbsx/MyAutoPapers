use crate::types::Paper;
use crate::utils::remove_duplicated_spaces;
use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use feed_rs::parser;
use url::Url;

const ARXIV_API_BASE_URL: &str = "https://export.arxiv.org/api/query";
const ARXIV_USER_AGENT: &str = "MyAutoPapers/0.1 (https://github.com/dbsxdbsx/MyAutoPapers)";
pub const ARXIV_REQUEST_INTERVAL_SECS: u64 = 5;

pub async fn request_paper_with_arxiv_api(
    keyword: &str,
    per_keyword_max_results: usize,
    link: &str,
) -> Result<Vec<Paper>> {
    let search_query = format!("ti:\"{keyword}\" {link} abs:\"{keyword}\"");
    let mut url = Url::parse(ARXIV_API_BASE_URL)?;
    url.query_pairs_mut()
        .append_pair("search_query", &search_query)
        .append_pair("max_results", &per_keyword_max_results.to_string())
        .append_pair("sortBy", "lastUpdatedDate");

    println!("正在从 arXiv 获取论文数据...,url:{url}");

    let client = reqwest::Client::builder()
        .user_agent(ARXIV_USER_AGENT)
        .build()
        .context("创建 arXiv HTTP client 失败")?;

    println!("正在从 arXiv 获取论文数据...");
    let response = client
        .get(url.clone())
        .send()
        .await
        .with_context(|| format!("请求 arXiv API 失败: {url}"))?;
    let status = response.status();
    let response_text = response
        .text()
        .await
        .with_context(|| format!("读取 arXiv API 响应失败: {url}"))?;

    if !status.is_success() {
        let preview = response_text.chars().take(200).collect::<String>();
        bail!("arXiv API 返回非成功状态 {status}，响应前缀: {preview:?}");
    }

    if response_text.trim().is_empty() {
        bail!("arXiv API 返回空响应");
    }

    if !response_text.trim_start().starts_with("<?xml")
        && !response_text.trim_start().starts_with("<feed")
    {
        let preview = response_text.chars().take(200).collect::<String>();
        bail!("arXiv API 返回内容不是 Atom/XML feed，响应前缀: {preview:?}");
    }

    let feed = parser::parse(response_text.as_bytes()).context("解析 arXiv Atom feed 失败")?;

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

pub async fn get_filtered_papers_by_keyword_with_retries(
    keyword: &str,
    retry_times: usize,
    per_keyword_max_result: usize,
    link: &str,
    exclude_keywords: &[String],
) -> Result<Vec<Paper>> {
    let retry_times = retry_times.max(3);
    let mut last_error = None;
    let mut saw_successful_empty_result = false;

    for i in 0..retry_times {
        match get_daily_papers_by_keyword(keyword, per_keyword_max_result, link).await {
            Ok(papers) if papers.is_empty() => {
                saw_successful_empty_result = true;
                println!(
                    "第 {} 次尝试最终结果为 0 篇，{} 次以内会继续重试...",
                    i + 1,
                    retry_times
                );
            }
            Ok(papers) => {
                let filtered = filter_papers(papers, exclude_keywords);
                if filtered.is_empty() {
                    saw_successful_empty_result = true;
                    println!(
                        "第 {} 次尝试过滤后结果为 0 篇，{} 次以内会继续重试...",
                        i + 1,
                        retry_times
                    );
                } else {
                    return Ok(filtered);
                }
            }
            Err(e) => {
                println!("第 {} 次尝试失败: {}，重试中...", i + 1, e);
                last_error = Some(e);
            }
        }

        if i + 1 < retry_times {
            tokio::time::sleep(tokio::time::Duration::from_secs(
                ARXIV_REQUEST_INTERVAL_SECS,
            ))
            .await;
        }
    }

    if saw_successful_empty_result {
        return Ok(Vec::new());
    }

    if let Some(error) = last_error {
        Err(anyhow!(
            "尝试 {} 次后仍未成功获取非空结果，最后一次错误: {error}",
            retry_times
        ))
    } else {
        Ok(Vec::new())
    }
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

    let mut failed_sub_keywords = Vec::new();
    let mut succeeded_sub_keyword_count = 0;

    for (index, sub_key) in sub_keywords.iter().enumerate() {
        println!("-> 处理子关键词: {sub_key}");
        match request_paper_with_arxiv_api(sub_key, per_keyword_max_result, link).await {
            Ok(mut papers) => {
                succeeded_sub_keyword_count += 1;
                all_papers.append(&mut papers);
            }
            Err(error) => {
                println!("子关键词 '{sub_key}' 拉取失败: {error}");
                failed_sub_keywords.push(format!("{sub_key}: {error}"));
            }
        }

        if index + 1 < sub_keywords.len() {
            tokio::time::sleep(tokio::time::Duration::from_secs(
                ARXIV_REQUEST_INTERVAL_SECS,
            ))
            .await;
        }
    }

    if succeeded_sub_keyword_count == 0 && !failed_sub_keywords.is_empty() {
        bail!(
            "所有子关键词请求都失败: {}",
            failed_sub_keywords.join(" | ")
        );
    }

    if !failed_sub_keywords.is_empty() {
        println!(
            "部分子关键词失败，保留已成功结果: {}",
            failed_sub_keywords.join(" | ")
        );
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
