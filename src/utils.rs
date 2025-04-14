use crate::types::Paper;
use chrono::Utc;
use chrono_tz::Asia::Shanghai;
use regex::Regex;
use std::fs;
use std::path::Path;

pub fn remove_duplicated_spaces(text: &str) -> String {
    let re = Regex::new(r"\s+").unwrap();
    re.replace_all(text.trim(), " ").to_string()
}

// 获取当前日期，用于每月更新的标题
pub fn get_daily_date() -> String {
    Utc::now()
        .with_timezone(&Shanghai)
        .format("%Y年%m月%d日")
        .to_string()
}

pub fn get_current_date_time() -> String {
    Utc::now()
        .with_timezone(&Shanghai)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

pub fn generate_table(papers: &[Paper], context: &str, original_keyword: &str) -> String {
    if papers.is_empty() {
        return format!("*未找到与'{original_keyword}'相关的新论文*");
    }

    let header = Paper::markdown_header(context);
    let rows: Vec<String> = papers
        .iter()
        .enumerate()
        .map(|(i, paper)| match context {
            "readme" => paper.to_readme_markdown(i),
            "issue" => paper.to_issue_markdown(i),
            _ => String::new(),
        })
        .collect();

    format!("{}\n{}", header, rows.join("\n"))
}

pub fn ensure_github_dir() {
    let github_dir = Path::new(".github");
    if !github_dir.exists() {
        fs::create_dir_all(github_dir).expect("创建 .github 目录失败");
    }
}

pub fn backup_files() -> anyhow::Result<()> {
    if Path::new("README.md").exists() {
        fs::copy("README.md", "README.md.bk")?;
    }
    if Path::new(".github/ISSUE_TEMPLATE.md").exists() {
        fs::copy(".github/ISSUE_TEMPLATE.md", ".github/ISSUE_TEMPLATE.md.bk")?;
    }
    Ok(())
}

pub fn remove_backups() -> anyhow::Result<()> {
    if Path::new("README.md.bk").exists() {
        fs::remove_file("README.md.bk")?;
    }
    if Path::new(".github/ISSUE_TEMPLATE.md.bk").exists() {
        fs::remove_file(".github/ISSUE_TEMPLATE.md.bk")?;
    }
    Ok(())
}
