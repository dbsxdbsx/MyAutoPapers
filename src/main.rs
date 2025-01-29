mod arxiv;
mod types;
mod utils;

use anyhow::Result;
use chrono::Utc;
use std::env;
use std::fs::File;
use std::io::Write;
use types::Config;
use utils::{
    backup_files, ensure_github_dir, generate_table, get_current_date_time, get_daily_date,
    remove_backups,
};

fn format_this_time_update_config_output(config: &Config) -> String {
    format!(
        "## 最后更新：{date}\n\
        **本次更新执行命令**\n\
        ```\n{command}\n```\n\n\
        **参数详解**\n\
        - 关键词：{keywords}\n\
        - 排除关键词：{exclude}\n\
        - 每关键词最大结果：`{per_keyword_max_result}`\n\
        - 目标领域：{target_fields}\n\
        - 每关键词重试次数：`{retry_times}`\n",
        command = env::args().collect::<Vec<_>>().join(" "),
        keywords = config
            .keywords
            .iter()
            .map(|k| format!("`{k}`"))
            .collect::<Vec<_>>()
            .join(", "),
        exclude = if config.exclude_keywords.is_empty() {
            "`(无)`".to_string()
        } else {
            config
                .exclude_keywords
                .iter()
                .map(|k| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ")
        },
        per_keyword_max_result = config.per_keyword_max_result,
        target_fields = config
            .target_fields
            .iter()
            .map(|k| format!("`{k}`"))
            .collect::<Vec<_>>()
            .join(", "),
        retry_times = config.retry_times_for_each_keyword,
        date = get_current_date_time()
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("开始运行论文抓取程序...");
    let args: Vec<String> = env::args().collect();
    let mut final_recorded_papers_num = 0;

    let config = Config {
        keywords: args
            .iter()
            .find(|a| a.starts_with("--keywords="))
            .map(|s| {
                let keywords = s
                    .split_once('=')
                    .unwrap()
                    .1
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect::<Vec<String>>();
                assert!(
                    !keywords.is_empty(),
                    "关键词参数 --keywords 不能为空，请提供至少一个搜索关键词"
                );
                keywords
            })
            .unwrap(),

        retry_times_for_each_keyword: args
            .iter()
            .find(|a| a.starts_with("--retry-times="))
            .and_then(|s| s.split_once('=').unwrap().1.parse().ok())
            .unwrap_or(3),

        target_fields: args
            .iter()
            .find(|a| a.starts_with("--target-fields="))
            .map(|s| {
                s.split_once('=')
                    .unwrap()
                    .1
                    .split(',')
                    .map(str::trim)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_else(|| vec!["cs".into(), "stat".into()]),

        exclude_keywords: args
            .iter()
            .find(|a| a.starts_with("--exclude-keywords="))
            .map(|s| {
                s.split_once('=')
                    .unwrap()
                    .1
                    .split(',')
                    .map(str::trim)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_else(|| vec!["multi-agent".into()]),

        per_keyword_max_result: args
            .iter()
            .find(|a| a.starts_with("--per-keyword-max-result="))
            .and_then(|s| s.split_once('=').unwrap().1.parse().ok())
            .unwrap_or(100),

        column_names_to_search: vec![
            "Title".to_string(),    // 标题
            "Date".to_string(),     // 日期
            "Abstract".to_string(), // 摘要
        ],
    };

    assert!(
        !config.target_fields.is_empty(),
        "必须至少指定一个目标领域（例如 --target-fields=cs,stat）"
    );

    println!("当前排除关键词: {:?}", config.exclude_keywords);

    println!("备份现有文件...");
    backup_files()?;

    println!("创建目录结构...");
    ensure_github_dir();

    println!("准备搜索并写入文件...");
    let mut readme = File::create("README.md")?;
    let mut issue_template = File::create(".github/ISSUE_TEMPLATE.md")?;

    // 先写入 issue 模板的头部信息
    writeln!(
        issue_template,
        "---\ntitle: 最新论文 - {date}\nlabels: documentation\n---\n\
        {params}\n\n\
        ## 论文汇总（{papers_num}篇）\n\n\
        **更好的阅读体验请访问 [Github页面](https://github.com/dbsxdbsx/MyDailyPaper)。**\n\n",
        date = get_daily_date(),
        params = format_this_time_update_config_output(&config),
        papers_num = 0 // 初始为0，稍后更新
    )?;

    // 先写入 README 的头部信息
    writeln!(
        readme,
        "# 自动论文推送\n\
        本项目自动从 arXiv 获取最新的论文，基于关键词进行筛选。\n\n\
        点击 'Watch' 按钮可以接收自动推送的邮件通知。\n\n\
        {update_info}\n\n",
        update_info = format_this_time_update_config_output(&config)
    )?;

    let mut all_display_papers = Vec::new();
    let keywords_len = config.keywords.len();
    let mut paper_contents = String::new();

    for (i, keyword) in config.keywords.iter().enumerate() {
        println!("\n处理组合关键词: {keyword}");
        paper_contents.push_str(&format!("### {}. {}\n", i + 1, keyword));

        let link = if keyword.split_whitespace().count() == 1 {
            "AND"
        } else {
            "OR"
        };

        match arxiv::get_daily_papers_by_keyword_with_retries(
            keyword,
            config.retry_times_for_each_keyword,
            config.per_keyword_max_result,
            link,
        )
        .await
        {
            Ok(papers) if !papers.is_empty() => {
                let mut filtered = arxiv::filter_papers(papers, &config.exclude_keywords);
                filtered.sort_by(|a, b| {
                    b.date_time()
                        .unwrap_or(Utc::now())
                        .cmp(&a.date_time().unwrap_or(Utc::now()))
                });

                let readme_table = generate_table(&filtered, "readme", keyword);
                let posted_issue_table = generate_table(&filtered, "issue", keyword);

                all_display_papers.extend(filtered.clone());

                paper_contents.push_str(&format!("{readme_table}\n"));
                writeln!(
                    issue_template,
                    "### {}. {}\n{}",
                    i + 1,
                    keyword,
                    posted_issue_table
                )?;

                final_recorded_papers_num += filtered.len();
                println!("过滤后：成功写入 {} 篇论文", filtered.len());
            }
            Ok(_) => {
                println!("未获取到论文，跳过处理");
                continue;
            }
            Err(e) => {
                eprintln!("处理关键词 '{keyword}' 时出错: {e}");
                continue;
            }
        }

        if i < keywords_len - 1 {
            let time_to_sleep = 1;
            println!(
                "等待 {time_to_sleep} 秒以避免 API 限制...\n----------------------------------"
            );
            tokio::time::sleep(tokio::time::Duration::from_secs(time_to_sleep)).await;
        }
    }

    // 在处理完所有论文后，更新 issue 模板的论文数量
    let issue_content = std::fs::read_to_string(".github/ISSUE_TEMPLATE.md")?;
    let updated_issue_content = issue_content
        .replace(
            "最新论文0篇",
            &format!("最新论文{final_recorded_papers_num}篇"),
        )
        .replace(
            "论文汇总（0篇）",
            &format!("论文汇总（{final_recorded_papers_num}篇）"),
        );
    std::fs::write(".github/ISSUE_TEMPLATE.md", updated_issue_content)?;

    // 写入 README 的论文内容
    writeln!(
        readme,
        "## 论文汇总（{final_recorded_papers_num}篇）\n\n{paper_contents}"
    )?;

    writeln!(
        readme,
        "\n# 鸣谢\n\
        感谢原始项目 [@zezhishao/DailyArXiv](https://github.com/zezhishao/DailyArXiv) 提供的灵感。"
    )?;

    println!("清理备份文件...");
    remove_backups()?;

    println!("程序执行完成！");
    Ok(())
}
