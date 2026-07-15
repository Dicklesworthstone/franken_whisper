use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq)]
struct VideoRef {
    id: String,
    title: String,
    url: String,
    duration_sec: Option<f64>,
}

#[derive(serde::Deserialize)]
struct FlatPlaylistEntry {
    #[serde(default)]
    id: serde_json::Value,
    #[serde(default)]
    title: serde_json::Value,
    #[serde(default)]
    url: serde_json::Value,
    #[serde(default)]
    webpage_url: serde_json::Value,
    #[serde(default)]
    duration: serde_json::Value,
}

fn non_empty_json_string(value: serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

fn parse_projected(line: &str) -> serde_json::Result<Option<VideoRef>> {
    serde_json::from_str::<FlatPlaylistEntry>(line).map(|entry| {
        let id = non_empty_json_string(entry.id)?;
        let title = non_empty_json_string(entry.title).unwrap_or_default();
        let url = non_empty_json_string(entry.url)
            .or_else(|| non_empty_json_string(entry.webpage_url))
            .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));
        Some(VideoRef {
            id,
            title,
            url,
            duration_sec: entry.duration.as_f64(),
        })
    })
}

fn string_field(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
}

fn parse_dom(line: &str) -> serde_json::Result<Option<VideoRef>> {
    serde_json::from_str::<serde_json::Value>(line).map(|value| {
        let id = value.get("id").and_then(serde_json::Value::as_str)?;
        if id.is_empty() {
            return None;
        }
        let title = string_field(&value, "title").unwrap_or_default();
        let url = string_field(&value, "url")
            .or_else(|| string_field(&value, "webpage_url"))
            .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));
        Some(VideoRef {
            id: id.to_owned(),
            title,
            url,
            duration_sec: value.get("duration").and_then(serde_json::Value::as_f64),
        })
    })
}

fn assert_exact(left: &Option<VideoRef>, right: &Option<VideoRef>) {
    match (left, right) {
        (Some(left), Some(right)) => {
            assert_eq!(left.id, right.id);
            assert_eq!(left.title, right.title);
            assert_eq!(left.url, right.url);
            assert_eq!(
                left.duration_sec.map(f64::to_bits),
                right.duration_sec.map(f64::to_bits)
            );
        }
        (None, None) => {}
        _ => panic!("DOM/projected mismatch: {left:?} != {right:?}"),
    }
}

fn parity_oracle() {
    for line in [
        r#"{"id":"abc","title":"title","url":"https://youtu.be/abc","webpage_url":"ignored","duration":1.25,"description":{"fat":[1,2,3]}}"#,
        r#"{"id":"escaped\\\"id","title":"snowman ☃","webpage_url":"https://example.test/watch","duration":-0.0}"#,
        r#"{"id":"abc","title":"","url":"","webpage_url":"","duration":7}"#,
        r#"{"id":"abc","title":7,"url":false,"webpage_url":"fallback","duration":"9"}"#,
        r#"{"id":""}"#,
        r#"{"id":42}"#,
        r#"{"title":"missing id"}"#,
    ] {
        assert_exact(&parse_dom(line).unwrap(), &parse_projected(line).unwrap());
    }
    for line in [r#"{"id":"unterminated""#, "{"] {
        assert!(parse_dom(line).is_err());
        assert!(parse_projected(line).is_err());
    }
}

fn realistic_lines(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| {
            serde_json::json!({
                "_type": "url",
                "ie_key": "Youtube",
                "id": format!("vid{index:08}xyz"),
                "url": format!("https://www.youtube.com/watch?v=vid{index:08}xyz"),
                "title": format!("A Reasonably Long Representative Playlist Video Title Number {index}"),
                "description": "A multi-sentence description that yt-dlp includes in flat dumps. It is typically a couple hundred characters of prose padding.",
                "duration": 245.0 + (index % 600) as f64,
                "channel_id": "UCabcdefghijklmnopqrstuv",
                "channel": "Some Representative Channel Name",
                "channel_url": "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv",
                "uploader": "Some Representative Channel Name",
                "uploader_id": "@somerepresentativechannel",
                "uploader_url": "https://www.youtube.com/@somerepresentativechannel",
                "view_count": 123456 + index,
                "availability": "public",
                "live_status": "not_live",
                "thumbnails": (0..5).map(|thumbnail| serde_json::json!({
                    "url": format!("https://i.ytimg.com/vi/vid{index:08}xyz/hqdefault_{thumbnail}.jpg"),
                    "height": 94 + thumbnail * 100,
                    "width": 168 + thumbnail * 160,
                })).collect::<Vec<_>>(),
            })
            .to_string()
        })
        .collect()
}

fn parse_signature(lines: &[String], projected: bool, rounds: usize) -> u64 {
    let mut signature = 0_u64;
    for _ in 0..rounds {
        for line in lines {
            let video = if projected {
                parse_projected(black_box(line))
            } else {
                parse_dom(black_box(line))
            }
            .unwrap()
            .unwrap();
            signature = signature
                .wrapping_mul(31)
                .wrapping_add(video.id.len() as u64)
                .wrapping_add(video.title.len() as u64)
                .wrapping_add(video.url.len() as u64)
                .wrapping_add(video.duration_sec.unwrap().to_bits());
        }
    }
    black_box(signature)
}

fn timed(lines: &[String], projected: bool, rounds: usize) -> (Duration, u64) {
    let started = Instant::now();
    let signature = parse_signature(lines, projected, rounds);
    (started.elapsed(), signature)
}

fn paired_ratios(lines: &[String], projected_second: bool) -> Vec<f64> {
    const PAIRS: usize = 15;
    const ROUNDS_PER_ARM: usize = 32;
    let mut ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let (first_before, second_before, second_after, first_after) = if pair % 2 == 0 {
            let first_before = timed(lines, false, ROUNDS_PER_ARM);
            let second_before = timed(lines, projected_second, ROUNDS_PER_ARM);
            let second_after = timed(lines, projected_second, ROUNDS_PER_ARM);
            let first_after = timed(lines, false, ROUNDS_PER_ARM);
            (first_before, second_before, second_after, first_after)
        } else {
            let second_before = timed(lines, projected_second, ROUNDS_PER_ARM);
            let first_before = timed(lines, false, ROUNDS_PER_ARM);
            let first_after = timed(lines, false, ROUNDS_PER_ARM);
            let second_after = timed(lines, projected_second, ROUNDS_PER_ARM);
            (first_before, second_before, second_after, first_after)
        };
        assert_eq!(first_before.1, second_before.1);
        assert_eq!(first_after.1, second_after.1);
        ratios.push(
            (first_before.0 + first_after.0).as_secs_f64()
                / (second_before.0 + second_after.0).as_secs_f64(),
        );
    }
    ratios
}

fn percentile(ratios: &[f64], percent: usize) -> f64 {
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (sorted.len() * percent).div_ceil(100).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn cv_percent(ratios: &[f64]) -> f64 {
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let variance = ratios
        .iter()
        .map(|ratio| (ratio - mean).powi(2))
        .sum::<f64>()
        / (ratios.len() - 1) as f64;
    variance.sqrt() / mean * 100.0
}

fn main() {
    parity_oracle();
    let lines = realistic_lines(128);
    let bytes = lines.iter().map(String::len).sum::<usize>();
    for _ in 0..3 {
        assert_eq!(
            parse_signature(&lines, false, 4),
            parse_signature(&lines, true, 4)
        );
    }

    let null = paired_ratios(&lines, false);
    let candidate = paired_ratios(&lines, true);
    let null_median = percentile(&null, 50);
    let null_p90 = percentile(&null, 90);
    let candidate_p10 = percentile(&candidate, 10);
    let candidate_median = percentile(&candidate, 50);
    let candidate_wins = candidate.iter().filter(|&&ratio| ratio > 1.0).count();
    println!("FIXTURE_LINES=128 FIXTURE_BYTES={bytes} PARSES_PER_ARM=4096");
    println!("BASE_BASE_RATIOS={null:?}");
    println!("BASE_CANDIDATE_RATIOS={candidate:?}");
    println!(
        "NULL_MEDIAN={null_median:.6} NULL_P90={null_p90:.6} NULL_CV_PCT={:.3} CANDIDATE_P10={candidate_p10:.6} CANDIDATE_MEDIAN={candidate_median:.6} CANDIDATE_CV_PCT={:.3} CANDIDATE_WINS={candidate_wins}/15",
        cv_percent(&null),
        cv_percent(&candidate)
    );

    assert!((0.98..=1.02).contains(&null_median));
    assert!(candidate_median >= 1.20);
    assert!(candidate_p10 > null_p90);
    assert!(candidate_wins >= 13);
}
