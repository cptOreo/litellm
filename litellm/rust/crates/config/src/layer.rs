//! The environment as a layer under config.yaml. A key the config leaves unset and a variable
//! backs is read from that variable here, once, so nothing past this crate asks which of the two
//! set it.
//!
//! One rule for every such key: the config's own value first, then `LITELLM_<KEY>`, then the
//! default. The proxy reads some of these variables only when it starts, before the config's
//! `environment_variables` are exported, and reads a word that is not a flag as on. Neither is
//! kept: the variable is read from whatever `environment` is given, and its text is read as the
//! key's own text would be.

use std::collections::BTreeMap;

use litellm_env::Lookup;

use crate::{Config, Spelled};

/// Which config keys the environment set, each with the variable it was read from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Origins(BTreeMap<&'static str, String>);

impl Origins {
    /// The variable that set the config key `key`, as `litellm_settings.drop_params` names one,
    /// when the environment set it.
    pub fn variable(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }
}

/// What a key holds when its variable set it: the variable's text, as the key's own text would
/// be read.
trait Written {
    fn written(text: String) -> Self;
}

impl<T> Written for Spelled<T> {
    fn written(text: String) -> Self {
        Spelled::Text(text)
    }
}

impl Written for String {
    fn written(text: String) -> Self {
        text
    }
}

/// `$settings` with each of the fields it leaves unset read from `LITELLM_<FIELD>`, and the key
/// and variable of each one that was. A blank variable is unset.
macro_rules! layered {
    ($settings:expr, $environment:expr, $section:literal, [$($field:ident),+ $(,)?]) => {{
        let mut settings = $settings;
        let mut origins = Vec::new();
        $(
            let variable = concat!("LITELLM_", stringify!($field)).to_uppercase();
            let written = $environment
                .get(&variable)
                .filter(|text| !text.trim().is_empty());
            match written {
                Some(text) if settings.$field.is_none() => {
                    settings.$field = Some(Written::written(text));
                    origins.push((concat!($section, ".", stringify!($field)), variable));
                }
                _ => {}
            }
        )+
        (settings, origins)
    }};
}

impl Config {
    /// This config with `environment` under it.
    pub fn with_environment(self, environment: &dyn Lookup) -> Self {
        let (litellm_settings, origins) = layered!(
            self.litellm_settings,
            environment,
            "litellm_settings",
            [
                drop_params,
                modify_params,
                enable_anthropic_prompt_caching,
                reasoning_auto_summary,
                use_chat_completions_url_for_anthropic_messages,
                route_all_chat_openai_to_responses,
                gemini_live_defer_setup,
                anthropic_prompt_caching_ttl,
                model_cost_map_url,
                blog_posts_url,
                anthropic_beta_headers_url,
                autorouter_presets_url,
            ]
        );
        Self {
            litellm_settings,
            origins: Origins(origins.into_iter().collect()),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const KEY: &str = "litellm_settings.drop_params";

    fn layered(yaml: &str, written: Option<&str>) -> Config {
        let lookup = |name: &str| {
            (name == "LITELLM_DROP_PARAMS")
                .then_some(written)
                .flatten()
                .map(str::to_owned)
        };
        Config::from_yaml(yaml).unwrap().with_environment(&lookup)
    }

    #[rstest]
    #[case::the_variable_when_the_config_leaves_the_key_unset(
        "model_list: []\n",
        Some("true"),
        Some(Spelled::Text("true".into())),
        Some("LITELLM_DROP_PARAMS")
    )]
    #[case::the_config_when_both_set_it(
        "litellm_settings:\n  drop_params: false\n",
        Some("true"),
        Some(Spelled::Value(false)),
        None
    )]
    #[case::neither("model_list: []\n", None, None, None)]
    #[case::a_blank_variable("model_list: []\n", Some("  "), None, None)]
    fn a_key_is_read_from_the_config_first_and_its_variable_second(
        #[case] yaml: &str,
        #[case] written: Option<&str>,
        #[case] value: Option<Spelled<bool>>,
        #[case] variable: Option<&str>,
    ) {
        let config = layered(yaml, written);

        assert_eq!(
            (
                config.litellm_settings.drop_params,
                config.origins.variable(KEY)
            ),
            (value, variable)
        );
    }

    #[rstest]
    fn a_variable_is_named_after_its_key() {
        let lookup = |name: &str| {
            (name == "LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES")
                .then(|| "true".to_owned())
        };

        let config = Config::default().with_environment(&lookup);

        assert_eq!(
            config
                .litellm_settings
                .use_chat_completions_url_for_anthropic_messages,
            Some(Spelled::Text("true".into()))
        );
    }

    #[rstest]
    fn a_text_key_holds_its_variables_text() {
        let lookup =
            |name: &str| (name == "LITELLM_ANTHROPIC_PROMPT_CACHING_TTL").then(|| "1h".to_owned());

        let config = Config::default().with_environment(&lookup);

        assert_eq!(
            (
                config
                    .litellm_settings
                    .anthropic_prompt_caching_ttl
                    .as_deref(),
                config
                    .origins
                    .variable("litellm_settings.anthropic_prompt_caching_ttl")
            ),
            (Some("1h"), Some("LITELLM_ANTHROPIC_PROMPT_CACHING_TTL"))
        );
    }
}
