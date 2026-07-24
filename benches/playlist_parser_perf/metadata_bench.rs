use std::fmt;
use std::hint::black_box;
use std::time::{Duration, Instant};

use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};

#[derive(Debug, PartialEq)]
struct VideoMeta {
    id: String,
    title: String,
    channel: Option<String>,
    uploader: Option<String>,
    upload_date: Option<String>,
    duration_sec: Option<f64>,
    webpage_url: String,
    description: Option<String>,
    availability: Option<String>,
    live_status: Option<String>,
}

struct ProjectedMetadata {
    id: serde_json::Value,
    title: serde_json::Value,
    channel: serde_json::Value,
    uploader: serde_json::Value,
    upload_date: serde_json::Value,
    duration: serde_json::Value,
    webpage_url: serde_json::Value,
    description: serde_json::Value,
    availability: serde_json::Value,
    live_status: serde_json::Value,
}

impl Default for ProjectedMetadata {
    fn default() -> Self {
        Self {
            id: serde_json::Value::Null,
            title: serde_json::Value::Null,
            channel: serde_json::Value::Null,
            uploader: serde_json::Value::Null,
            upload_date: serde_json::Value::Null,
            duration: serde_json::Value::Null,
            webpage_url: serde_json::Value::Null,
            description: serde_json::Value::Null,
            availability: serde_json::Value::Null,
            live_status: serde_json::Value::Null,
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum MetadataField {
    Id,
    Title,
    Channel,
    Uploader,
    UploadDate,
    Duration,
    WebpageUrl,
    Description,
    Availability,
    LiveStatus,
    #[serde(other)]
    Other,
}

impl<'de> serde::Deserialize<'de> for ProjectedMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MetadataVisitor;

        impl<'de> Visitor<'de> for MetadataVisitor {
            type Value = ProjectedMetadata;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a yt-dlp metadata JSON value")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut metadata = ProjectedMetadata::default();
                while let Some(field) = map.next_key()? {
                    match field {
                        MetadataField::Id => metadata.id = map.next_value()?,
                        MetadataField::Title => metadata.title = map.next_value()?,
                        MetadataField::Channel => metadata.channel = map.next_value()?,
                        MetadataField::Uploader => metadata.uploader = map.next_value()?,
                        MetadataField::UploadDate => metadata.upload_date = map.next_value()?,
                        MetadataField::Duration => metadata.duration = map.next_value()?,
                        MetadataField::WebpageUrl => metadata.webpage_url = map.next_value()?,
                        MetadataField::Description => metadata.description = map.next_value()?,
                        MetadataField::Availability => metadata.availability = map.next_value()?,
                        MetadataField::LiveStatus => metadata.live_status = map.next_value()?,
                        MetadataField::Other => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(metadata)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while sequence.next_element::<IgnoredAny>()?.is_some() {}
                Ok(ProjectedMetadata::default())
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }

            fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(ProjectedMetadata::default())
            }
        }

        deserializer.deserialize_any(MetadataVisitor)
    }
}

#[derive(Debug, PartialEq)]
enum ParseError {
    Json,
    MissingId,
}

fn non_empty_json_string(value: serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

fn string_field(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
}

fn parse_dom(line: &str) -> Result<VideoMeta, ParseError> {
    let value: serde_json::Value = serde_json::from_str(line).map_err(|_| ParseError::Json)?;
    let id = string_field(&value, "id").ok_or(ParseError::MissingId)?;
    let title = string_field(&value, "title").unwrap_or_default();
    let webpage_url = string_field(&value, "webpage_url")
        .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));
    Ok(VideoMeta {
        id,
        title,
        channel: string_field(&value, "channel"),
        uploader: string_field(&value, "uploader"),
        upload_date: string_field(&value, "upload_date"),
        duration_sec: value.get("duration").and_then(serde_json::Value::as_f64),
        webpage_url,
        description: string_field(&value, "description"),
        availability: string_field(&value, "availability"),
        live_status: string_field(&value, "live_status"),
    })
}

fn parse_projected(line: &str) -> Result<VideoMeta, ParseError> {
    let metadata: ProjectedMetadata = serde_json::from_str(line).map_err(|_| ParseError::Json)?;
    let id = non_empty_json_string(metadata.id).ok_or(ParseError::MissingId)?;
    let title = non_empty_json_string(metadata.title).unwrap_or_default();
    let webpage_url = non_empty_json_string(metadata.webpage_url)
        .unwrap_or_else(|| format!("https://www.youtube.com/watch?v={id}"));
    Ok(VideoMeta {
        id,
        title,
        channel: non_empty_json_string(metadata.channel),
        uploader: non_empty_json_string(metadata.uploader),
        upload_date: non_empty_json_string(metadata.upload_date),
        duration_sec: metadata.duration.as_f64(),
        webpage_url,
        description: non_empty_json_string(metadata.description),
        availability: non_empty_json_string(metadata.availability),
        live_status: non_empty_json_string(metadata.live_status),
    })
}

fn assert_exact(left: &Result<VideoMeta, ParseError>, right: &Result<VideoMeta, ParseError>) {
    match (left, right) {
        (Ok(left), Ok(right)) => {
            assert_eq!(left.id, right.id);
            assert_eq!(left.title, right.title);
            assert_eq!(left.channel, right.channel);
            assert_eq!(left.uploader, right.uploader);
            assert_eq!(left.upload_date, right.upload_date);
            assert_eq!(
                left.duration_sec.map(f64::to_bits),
                right.duration_sec.map(f64::to_bits)
            );
            assert_eq!(left.webpage_url, right.webpage_url);
            assert_eq!(left.description, right.description);
            assert_eq!(left.availability, right.availability);
            assert_eq!(left.live_status, right.live_status);
        }
        (Err(left), Err(right)) => assert_eq!(left, right),
        _ => panic!("DOM/projected mismatch: {left:?} != {right:?}"),
    }
}

fn parity_oracle() {
    for line in [
        r#"{"id":"abc","title":"title","channel":"channel","uploader":"uploader","upload_date":"20260102","duration":1.25,"webpage_url":"https://youtu.be/abc","description":"description","availability":"public","live_status":"not_live","ignored":{"fat":[1,2,3]}}"#,
        r#"{"id":"abc","title":"","channel":null,"uploader":7,"duration":-0.0,"webpage_url":""}"#,
        r#"{"id":42,"title":"wrong id type"}"#,
        r#"{"title":"missing id"}"#,
        r#"{"id":"first","id":"last","title":"last duplicate wins"}"#,
        r#"[1,{"id":"nested"}]"#,
        r#"null"#,
        r#"true"#,
        r#""scalar""#,
    ] {
        assert_exact(&parse_dom(line), &parse_projected(line));
    }
    for line in [r#"{"id":"unterminated""#, "{"] {
        assert_exact(&parse_dom(line), &parse_projected(line));
    }
}

fn realistic_lines(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| {
            serde_json::json!({
                "id": format!("vid{index:08}xyz"),
                "title": format!("Representative Full Metadata Video Title Number {index}"),
                "channel": "Some Representative Channel Name",
                "channel_id": "UCabcdefghijklmnopqrstuv",
                "uploader": "Some Representative Channel Name",
                "uploader_id": "@somerepresentativechannel",
                "upload_date": "20260715",
                "duration": 3_600.125 + index as f64,
                "webpage_url": format!("https://www.youtube.com/watch?v=vid{index:08}xyz"),
                "description": "A representative long-form description. ".repeat(64),
                "availability": "public",
                "live_status": "not_live",
                "view_count": 123_456_789_u64 + index as u64,
                "like_count": 2_345_678_u64 + index as u64,
                "comment_count": 54_321_u64 + index as u64,
                "categories": ["Science & Technology", "Education"],
                "tags": (0..48).map(|tag| format!("representative-tag-{tag}")).collect::<Vec<_>>(),
                "thumbnails": (0..12).map(|thumbnail| serde_json::json!({
                    "url": format!("https://i.ytimg.com/vi/vid{index:08}xyz/thumb-{thumbnail}.jpg"),
                    "height": 90 + thumbnail * 90,
                    "width": 160 + thumbnail * 160,
                    "preference": thumbnail,
                })).collect::<Vec<_>>(),
                "formats": (0..24).map(|format_index| serde_json::json!({
                    "format_id": format!("fmt-{format_index}"),
                    "url": format!("https://rr.example.test/videoplayback?id={index}&format={format_index}&token={}", "x".repeat(160)),
                    "ext": if format_index % 2 == 0 { "webm" } else { "mp4" },
                    "acodec": if format_index % 3 == 0 { "opus" } else { "none" },
                    "vcodec": if format_index % 3 == 0 { "none" } else { "av01.0.08M.08" },
                    "width": 640 + format_index * 32,
                    "height": 360 + format_index * 18,
                    "fps": 30.0,
                    "filesize_approx": 10_000_000_u64 + format_index as u64 * 1_000_000,
                    "http_headers": {
                        "User-Agent": "Mozilla/5.0 representative benchmark agent",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    }
                })).collect::<Vec<_>>(),
                "subtitles": {
                    "en": (0..4).map(|subtitle| serde_json::json!({
                        "ext": "vtt",
                        "url": format!("https://www.youtube.com/api/timedtext?id={index}&track={subtitle}&token={}", "s".repeat(128))
                    })).collect::<Vec<_>>()
                },
                "automatic_captions": {
                    "en": (0..4).map(|caption| serde_json::json!({
                        "ext": "json3",
                        "url": format!("https://www.youtube.com/api/timedtext?id={index}&auto={caption}&token={}", "c".repeat(128))
                    })).collect::<Vec<_>>()
                }
            })
            .to_string()
        })
        .collect()
}

fn optional_len(value: &Option<String>) -> u64 {
    value.as_ref().map_or(0, |value| value.len() as u64)
}

fn parse_signature(lines: &[String], projected: bool, rounds: usize) -> u64 {
    let mut signature = 0_u64;
    for _ in 0..rounds {
        for line in lines {
            let metadata = if projected {
                parse_projected(black_box(line))
            } else {
                parse_dom(black_box(line))
            }
            .unwrap();
            signature = signature
                .wrapping_mul(31)
                .wrapping_add(metadata.id.len() as u64)
                .wrapping_add(metadata.title.len() as u64)
                .wrapping_add(optional_len(&metadata.channel))
                .wrapping_add(optional_len(&metadata.uploader))
                .wrapping_add(optional_len(&metadata.upload_date))
                .wrapping_add(metadata.duration_sec.unwrap().to_bits())
                .wrapping_add(metadata.webpage_url.len() as u64)
                .wrapping_add(optional_len(&metadata.description))
                .wrapping_add(optional_len(&metadata.availability))
                .wrapping_add(optional_len(&metadata.live_status));
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
    const ROUNDS_PER_ARM: usize = 8;
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
    let lines = realistic_lines(24);
    let bytes = lines.iter().map(String::len).sum::<usize>();
    for _ in 0..3 {
        assert_eq!(
            parse_signature(&lines, false, 2),
            parse_signature(&lines, true, 2)
        );
    }

    let null = paired_ratios(&lines, false);
    let candidate = paired_ratios(&lines, true);
    let null_median = percentile(&null, 50);
    let null_p90 = percentile(&null, 90);
    let candidate_p10 = percentile(&candidate, 10);
    let candidate_median = percentile(&candidate, 50);
    let candidate_wins = candidate.iter().filter(|&&ratio| ratio > 1.0).count();
    println!("FIXTURE_OBJECTS=24 FIXTURE_BYTES={bytes} PARSES_PER_ARM=192");
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
