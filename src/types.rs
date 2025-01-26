use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paper {
    pub title: String,
    pub authors: Vec<String>,
    pub abstract_text: String,
    pub link: String,
    pub tags: Vec<String>,
    pub date: String,
}

impl Paper {
    // 新增方法：生成 README 表格行
    pub fn to_readme_markdown(&self, index: usize) -> String {
        let title_link = format!("**[{}]({})**", self.title, self.link);
        let date = self.date.split('T').next().unwrap_or("").to_string();
        let abstract_details = format!(
            "<details><summary>展开</summary><p>{}</p></details>",
            self.abstract_text
        );
        format!(
            "| **{}** | {} | {} | {} |",
            index + 1,
            title_link,
            date,
            abstract_details
        )
    }

    // 新增方法：生成 Issue 的详细表格行
    pub fn to_issue_markdown(&self, index: usize) -> String {
        let title_link = format!("**[{}]({})**", self.title, self.link);
        let date = self.date.split('T').next().unwrap_or("").to_string();
        format!("| **{}** | {} | {} |", index + 1, title_link, date)
    }

    // 新增方法：生成表格头（可复用）
    pub fn markdown_header(context: &str) -> String {
        match context {
            "readme" => "| **序号** | **标题** | **日期** | **摘要** |\n| --- | --- | --- | --- |"
                .to_string(),
            "issue" => "| **序号** | **标题** | **日期** |\n| --- | --- | --- |".to_string(),
            _ => String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub keywords: Vec<String>,
    pub exclude_keywords: Vec<String>,
    #[serde(default = "default_target_fields")]
    pub target_fields: Vec<String>,
    pub per_keyword_max_result: usize,
    pub column_names_to_search: Vec<String>,
    pub retry_times_for_each_keyword: usize,
}

fn default_target_fields() -> Vec<String> {
    vec!["cs".into(), "stat".into()]
}
