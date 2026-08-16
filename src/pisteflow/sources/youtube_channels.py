"""Metadata-only catalog for public sport-fencing YouTube channels."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from pydantic import HttpUrl

from pisteflow.models import LicenseStatus, SourceKind, VideoRecord
from pisteflow.provenance import read_jsonl, write_json, write_jsonl
from pisteflow.sources.classification import (
    classify_content,
    infer_weapon_candidates,
    primary_weapon,
)

# Every entry was resolved to this exact channel ID before inclusion. Publisher types
# describe the observed channel identity or role; they are not endorsements or media-rights
# grants. Public community, coaching, analysis, and archival channels remain link-only.
CHANNELS: tuple[dict[str, str], ...] = (
    {
        "id": "UC1oc3twlZObCWUh2rsYMdig",
        "name": "USA Fencing",
        "publisher_type": "official_federation",
    },
    {
        "id": "UCL6amgBVv7DS0fB_3gSWssQ",
        "name": "British Fencing",
        "publisher_type": "official_federation",
    },
    {
        "id": "UCjC3skzqXCykT08GLiTcKyg",
        "name": "CyrusofChaos",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UC4VInV1FuLmCwIzqlOaQi5g",
        "name": "IrishFencingVideos",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCoX1Sn9VkFT7mTFU9ojzh8Q",
        "name": "AJC Fencing",
        "publisher_type": "coaching_and_analysis",
    },
    {
        "id": "UCmtALz8yYLoSIy6LkXs-h0w",
        "name": "Olympic Foil",
        "publisher_type": "community_analysis",
    },
    {
        "id": "UCCT4FI5gT5_uXuUdWZ_M8MQ",
        "name": "Slicer Sabre",
        "publisher_type": "community_analysis",
    },
    {
        "id": "UC5w-Tat0I_I9015HScSkNPQ",
        "name": "Fencing Insider",
        "publisher_type": "community_analysis",
    },
    {
        "id": "UCkOr1KHn9w8VsfonPXMmL1Q",
        "name": "GP Fencing",
        "publisher_type": "community_analysis",
    },
    {
        "id": "UC0cYbI_OZLy4HfRnDFC55kQ",
        "name": "Fencing Database",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCJXmuJdEMLOGuFDcdFIZOTA",
        "name": "Fencing 4 Everyone",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCaRxWBNb8YwJUdVfwINAv2A",
        "name": "Dancing tuna",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCo_7mlR3rhPc9UClczYuArA",
        "name": "USJ Fencing Centre",
        "publisher_type": "club_bout_archive",
    },
    {
        "id": "UCh0Am2q80cN9VevgNz0Qs6g",
        "name": "Fencing Bouts",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCgoL4eiznN5bwOQgGpu20Gg",
        "name": "fencing touch",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCQfC8fyObtNHEAzLHUB2WAw",
        "name": "Wilfie",
        "publisher_type": "wheelchair_fencing_archive",
    },
    {
        "id": "UCu02NBwxhvz0JmmaeerZTbA",
        "name": "Direct Elimination",
        "publisher_type": "community_analysis",
    },
    {
        "id": "UCBxZKGL7muejcX1PqEAwBpA",
        "name": "Fencing Data Project",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCnh4TwlECD8PFsUbVae6_Xg",
        "name": "Folo Fencing",
        "publisher_type": "community_highlights",
    },
    {
        "id": "UCxo3MuBTS-sl4xIrrPf04JQ",
        "name": "Tyshler fencing multimedia",
        "publisher_type": "coaching_and_instruction",
    },
    {
        "id": "UCndKNCqgmw8sAvcYoRboevg",
        "name": "Fencer's Edge",
        "publisher_type": "coaching_and_instruction",
    },
    {
        "id": "UCg-YsbVp_L7TnuyMwz5nBmA",
        "name": "Big Bear Fencing",
        "publisher_type": "coaching_and_instruction",
    },
    {
        "id": "UC90kVAHCuzbTfTYcS9rPIAA",
        "name": "Eurofencing",
        "publisher_type": "continental_confederation_channel",
    },
    {
        "id": "UCtCZKi07Dt6Hz4WfiJTuqzw",
        "name": "FederSchermaTV 1",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCzd9rg55xDfX1JQOeBJgFPA",
        "name": "Australian Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCIbUWPGI3sixL_jCDHQuxvA",
        "name": "Canadian Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCwxdwrMpnDW_QdfTTnTp21w",
        "name": "Japan Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCr-LluDLviq2XfXuyKhh2oA",
        "name": "Fencing Association of Hong Kong, China",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UC5_fZPJQiTi4ZAsBv_6KgpQ",
        "name": "Hong Kong Fencing Association archive",
        "publisher_type": "national_federation_archive_channel",
    },
    {
        "id": "UChEy3MoA2-WaVz_KDo_9KKA",
        "name": "Polski Zwiazek Szermierczy",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UC3UjRxcULbc5KbImOIMYzKw",
        "name": "Magyar Vivo Szovetseg",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCn388g7U-aMEpYSSZJfDxmg",
        "name": "Russian Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCQFkQEkAaW2otD7DtAPH-wA",
        "name": "Swiss Fencing",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCkZyVebCnjKRijoO_gTlNEA",
        "name": "Svenska Faktforbundet",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCQcuAK13GyT77c98pxHZENg",
        "name": "Federatia Romana de Scrima",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCHxKyK1zQBQciyckiaZuXKQ",
        "name": "National Fencing Federation of Ukraine",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCBR_--_qePbVusy0LQ2vErg",
        "name": "Turkiye Eskrim Federasyonu",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCOF5N7MTaDzD7swgWIx64Gw",
        "name": "CBEsgrima TV",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCqLqFhSZMvGCVKgpw_eMwmw",
        "name": "Deutscher Fechter-Bund",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCg3SB89tj8AKGHjJNo_GkLg",
        "name": "Real Federacion Espanola de Esgrima",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCk4SHIUWIXFzk8n9Q51hVMw",
        "name": "Esgrima Portugal",
        "publisher_type": "national_federation_named_channel",
    },
    {
        "id": "UCD7PBMcyay7sOo1IuVO6f9Q",
        "name": "Israel Fencing Association",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCNtMQ0jlVjg_BKr4YYbPZaA",
        "name": "Asian Fencing Federation",
        "publisher_type": "continental_federation_named_channel",
    },
    {
        "id": "UCe6-m0gSI3T-CNHE2mRwE5g",
        "name": "Egyptian Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UC-plChyceyT_8gdxw8zI38Q",
        "name": "National Fencing Federation of Kazakhstan",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCFNs3W3SFy4UjDEsu1wX7Lg",
        "name": "Fencing Federation of Uzbekistan",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCy4EIathBPbGxhgO4sT_pGg",
        "name": "Commonwealth Fencing",
        "publisher_type": "international_fencing_organization_channel",
    },
    {
        "id": "UCA40s4GODjkaJ9JeM0fBcdg",
        "name": "Fencing Vision",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCtxgwVJRFzPSneODB3t-uDQ",
        "name": "The Fencing Line",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UC7Zgyu3m7dB5DBZ3JbgfQaQ",
        "name": "FencerVideo",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCEI3t0bZwNt1PiyWN3QFWOg",
        "name": "schlager7",
        "publisher_type": "community_historical_bout_archive",
    },
    {
        "id": "UCWr27zz23LxnDZVx-HA9IYw",
        "name": "fencing visualized project",
        "publisher_type": "community_analysis",
    },
    {
        "id": "UCSjeeZaP92mdOICSXsaGdiw",
        "name": "ESCRIME TV",
        "publisher_type": "community_historical_bout_archive",
    },
    {
        "id": "UCz0O8K_pM8r2_aRThPjbiHw",
        "name": "FENCING INDIA",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCPwFBzTAte5xufVaPXmtjqA",
        "name": "OSM Fencing",
        "publisher_type": "club_bout_archive",
    },
    {
        "id": "UCu53KkaLclRB6EtSakf1jAw",
        "name": "Fencing Training",
        "publisher_type": "coaching_and_instruction",
    },
    {
        "id": "UCzHI9iuvaORC0VGT_WtRgqQ",
        "name": "Tim Morehouse Fencing Club",
        "publisher_type": "club_coaching_and_instruction",
    },
    {
        "id": "UCqIhpy73mDXb11yKpJWu1Fw",
        "name": "Fekete Gabor",
        "publisher_type": "coaching_and_instruction",
    },
    {
        "id": "UCQ5Ae0PvWdNvTBi1ZiMv4Gw",
        "name": "The Face Of Fencing",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCGNRvQsXG3ttOwI07TAYVMQ",
        "name": "S-Class",
        "publisher_type": "coaching_and_analysis",
    },
    {
        "id": "UCImqyG3ONIIhY--ZKjkKTfg",
        "name": "Sydney Sabre",
        "publisher_type": "community_sabre_archive_and_analysis",
    },
    {
        "id": "UCrUx6rFTICes5-n1dQKqz_w",
        "name": "Federacion Chilena Esgrima",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCp62f8AnEtPKSHAaAUgKITQ",
        "name": "Fencing Association of India",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCk5Qkd_1W6sTWCoMhijR8jg",
        "name": "Faegteklubben Trekanten",
        "publisher_type": "club_bout_archive",
    },
    {
        "id": "UCRaP3mE_qta1deXfg57BrrQ",
        "name": "FECBA Esgrima",
        "publisher_type": "regional_federation_channel",
    },
    {
        "id": "UCksM03gZahOqfkIZTxxDt6g",
        "name": "Thailand Fencing",
        "publisher_type": "national_fencing_archive_channel",
    },
    {
        "id": "UCvZnbWqaV-2ZlEn5lpVp56A",
        "name": "Rwandan ParaFencing Federation",
        "publisher_type": "national_parafencing_channel",
    },
    {
        "id": "UCjZNLYJnhiTIegJGJjOpilw",
        "name": "TheFencingCoach",
        "publisher_type": "community_analysis",
    },
    {
        "id": "UCCg3dWhW1bRqPip-vwJZUNA",
        "name": "AllezJKL",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCICzLvJ9UTDFo-A0-yHpnfg",
        "name": "ESGRIMA MASTER ARGENTINA",
        "publisher_type": "veteran_bout_archive",
    },
    {
        "id": "UC2zfNH_oc8zo_nQpfWgX80g",
        "name": "EZKRIMA",
        "publisher_type": "club_and_regional_bout_archive",
    },
    {
        "id": "UCYaJ5eTUbAIb-3PgRavexqw",
        "name": "Fencing Videos",
        "publisher_type": "community_instruction_and_analysis",
    },
    {
        "id": "UCDgvG4BA-tAhYRl4ZGylpAw",
        "name": "Cespedes Fencing",
        "publisher_type": "club_bout_and_training_archive",
    },
    {
        "id": "UCLpYywew3YCt5v34kq8o_LQ",
        "name": "Average Sabre Guy",
        "publisher_type": "community_sabre_analysis",
    },
    {
        "id": "UChBnKfyPGdfdUPwlIK6BHYA",
        "name": "GeekyFence",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCkBXUmM8xodWjdSpmYeckVQ",
        "name": "The Fencing Referee",
        "publisher_type": "referee_instruction",
    },
    {
        "id": "UCnIcV8K1Mz3jFtk7zV9sZyQ",
        "name": "Oleg Knysh",
        "publisher_type": "athlete_bout_archive",
    },
    {
        "id": "UC1i6YxTWvRRRnD-TknrTMug",
        "name": "didoigts",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCNGsMSE0I2MrtxveTgEXDnA",
        "name": "bj000008",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCnT4XHzfEBEQ04J8irG9jAA",
        "name": "Esgrima Costa Rica",
        "publisher_type": "regional_bout_archive",
    },
    {
        "id": "UCOeoedm60_n3Vh6bmc36CEw",
        "name": "ClubEsgrimaCR",
        "publisher_type": "club_bout_archive",
    },
    {
        "id": "UCfx9H8zux3nEB99D_hrMRVw",
        "name": "Greek Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCy7-KkAwjGEzyrVPYfm7Xfg",
        "name": "Fencing Federation South Africa",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCpY3iplYLFLSusFLxYWLO1A",
        "name": "Qatar Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCTsC2NZuJUwGDoenKHn5kTA",
        "name": "Federacion Mexicana de Esgrima",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCp1eCBPU71FmZmnTpzWc7Qw",
        "name": "Federacion Venezolana de Esgrima",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCmFrjZNyZ_gXj_U5ot_Q_7A",
        "name": "Esgrima Paralimpica Venezuela",
        "publisher_type": "national_parafencing_archive_channel",
    },
    {
        "id": "UCUTz8LrJEQyYXojZBucWYjQ",
        "name": "Saudi Fencing Federation",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UCEFZc5jG09Mx0K6308cE09A",
        "name": "fencingksa",
        "publisher_type": "national_federation_archive_channel",
    },
    {
        "id": "UCHKDvroK1dobEvTO39oYszQ",
        "name": "Federation Algerienne d'Escrime",
        "publisher_type": "national_federation_channel",
    },
    {
        "id": "UC0KTgQQN3ChlxVFLrIpvwUw",
        "name": "852fencing",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UCGDK5z-CjRvVS1XUpB3q5Tg",
        "name": "FENCING SEASON",
        "publisher_type": "community_bout_archive",
    },
    {
        "id": "UC5yNQ8ArpHo4B3_UOc2vcfw",
        "name": "Ariston Fencing",
        "publisher_type": "club_bout_and_training_archive",
    },
    {
        "id": "UCR39z8FKa3e8RIbxesHczCQ",
        "name": "Fencing4You",
        "publisher_type": "community_bout_and_highlight_archive",
    },
    {
        "id": "UCptQYxkbyGKDSQS3QwF_gxQ",
        "name": "Sportbite Fencing",
        "publisher_type": "community_bout_and_athlete_archive",
    },
    {
        "id": "UCIGfHECHllg40-kOyNrcqZQ",
        "name": "NCAA Fencing",
        "publisher_type": "collegiate_bout_archive",
    },
    {
        "id": "UCs2lZbedcsVQDhINBsfa38A",
        "name": "SethSZL Fencing",
        "publisher_type": "athlete_bout_archive",
    },
    {
        "id": "UC4VGDTr6RjdxWPyIuCT6Q1g",
        "name": "Delft Fencing Club",
        "publisher_type": "club_bout_and_training_archive",
    },
    {
        "id": "UCZC3nuaL5hElEwoW4MV2BRQ",
        "name": "PSVBerlin Fechten",
        "publisher_type": "club_bout_archive",
    },
    {
        "id": "UCF5m8K2CZAVclOaRUr9KeRw",
        "name": "Fechten Hamm",
        "publisher_type": "club_bout_and_training_archive",
    },
    {
        "id": "UCMJZLDS5lStxc0BomJzvQbg",
        "name": "avantgardefechten",
        "publisher_type": "club_bout_archive",
    },
    {
        "id": "UCkjoRLapKIT2JqWd2epkITA",
        "name": "Nauka Szermierki",
        "publisher_type": "coaching_and_bout_archive",
    },
    {
        "id": "UC4yopT9ApVqHaBzO-98adJw",
        "name": "Aichi Fencing Association",
        "publisher_type": "regional_federation_channel",
    },
    {
        "id": "UC4dzSQBPi9_7EMYvoZrFHrQ",
        "name": "Ogaki Fencing Association",
        "publisher_type": "regional_bout_archive",
    },
    {
        "id": "UCDjc3JaQPm_tfMjUcb5N2xw",
        "name": "Taipei City Fencing Association",
        "publisher_type": "regional_federation_channel",
    },
    {
        "id": "UCg5PKfRbq0X3v4lme03_1KA",
        "name": "UP Fencing Team Archives",
        "publisher_type": "collegiate_bout_archive",
    },
    {
        "id": "UCA9euyn39qE7SZmmfbcnRgA",
        "name": "West Coast Fencing Archive",
        "publisher_type": "community_historical_bout_archive",
    },
    {
        "id": "UCkkVI6G-6aDEyFHAj84yxCQ",
        "name": "fencingchannel",
        "publisher_type": "community_historical_bout_archive",
    },
    {
        "id": "UChra14LNO5Q9QraEImjVc1A",
        "name": "fencing.lv",
        "publisher_type": "community_veteran_bout_archive",
    },
    {
        "id": "UCOaTdco7BRn6mGs2z8j0Qww",
        "name": "Fencing.Net",
        "publisher_type": "community_bout_and_instruction_archive",
    },
    {
        "id": "UCZBUusUwrWOgXzIADKdAFyA",
        "name": "esgrima tv",
        "publisher_type": "community_pan_american_bout_archive",
    },
    {
        "id": "UCaaTTwWczilpkiPuAn9759A",
        "name": "Royal Arts Fencing Academy",
        "publisher_type": "club_bout_and_training_archive",
    },
    {
        "id": "UC18dlYjDn2o-Hxn4HASA0ZA",
        "name": "maitredarmes1andre",
        "publisher_type": "community_historical_bout_archive",
    },
)

TARGETED_SEARCHES: tuple[dict[str, str], ...] = (
    {
        "id": "UCi8n36NkW2uCQSFZNiYtuMQ",
        "name": "Paralympic Games",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "wheelchair fencing",
    },
    {
        "id": "UCTl3QQTvqHFjurroKxexy2Q",
        "name": "Olympic Games",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCqZQlzSHbVJrwrn5XvzrzcA",
        "name": "NBC Sports",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCTnzFhDigJXhORjkBFAAihg",
        "name": "World Abilitysport",
        "publisher_type": "parasport_organization_channel_targeted_search",
        "query": "para fencing",
    },
    {
        "id": "UCm1b_pmEdETz7YrNHgacszA",
        "name": "Panam Sports",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCU8NwQG55wOwWBDQksQOr_A",
        "name": "KSOC competition video",
        "publisher_type": "multisport_competition_channel_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCYAzhARb2dwI3_R_72r8hbA",
        "name": "Sport en France",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "escrime",
    },
    {
        "id": "UCYbbpwosQ0a2d3ygPpruJ1w",
        "name": "NCAA Championships",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCiEPYsBLfS7XKXamfV8_yRg",
        "name": "CONADEMX",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UC56hPhH8vBJ6ZemGvJAAFzQ",
        "name": "Centro Caribe Sports Channel",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UCk-d0Adte2OXCvMNxv6stvw",
        "name": "Chile Olimpico",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UCADFFCsD8JQBa9RQmyisjpA",
        "name": "Time Brasil TV",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UCXFbaLrEfExEg9mDa5Va8Iw",
        "name": "Jairo Piedrahita Lopera",
        "publisher_type": "mixed_personal_channel_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UCnnVB67jD2IhnsfyTbW9gRQ",
        "name": "Khalid Albraidi",
        "publisher_type": "mixed_sports_channel_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCWoiHFley8hGy0ad_t-0O7w",
        "name": "Sports For All",
        "publisher_type": "multisport_channel_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCciLSQTyi6W0UJ8ptreovXQ",
        "name": "TVES",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UCN6a_dOPlShlBDCEng7MSxA",
        "name": "Teleantioquia Deportes",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UCHULsbHTWYQhQjaU6oTwlow",
        "name": "APSF TV Channel",
        "publisher_type": "official_parasport_channel_targeted_search",
        "query": "wheelchair fencing",
    },
    {
        "id": "UCJXmxIlpj9eb-bNLYOel0iQ",
        "name": "Olympics Korea",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "펜싱",
    },
    {
        "id": "UCtybqqaTj6Nx74Azdz1KrsA",
        "name": "tvN SPORTS",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "펜싱",
    },
    {
        "id": "UCDIB1DOwPPe58M2fHPyVVDA",
        "name": "All Sports Korea",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "펜싱",
    },
    {
        "id": "UCqJi4dl1tZGJxmf2u_AHDXg",
        "name": "SmileSports",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "펜싱",
    },
    {
        "id": "UCg73Psdr5eCMk6KWFWeSiIw",
        "name": "Taiwan Sports",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "擊劍",
    },
    {
        "id": "UC1k-fvQOa6jgBsA6A3FGaZg",
        "name": "Taiwan Sports 2",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "擊劍",
    },
    {
        "id": "UCEenIlCU009lS0TZ8EKuKMw",
        "name": "HOP Sports",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "擊劍",
    },
    {
        "id": "UCHTK-2W11Vh1V4uwofOfR4w",
        "name": "AP Archive",
        "publisher_type": "historical_news_archive_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCEnl2ZDocvt4aoIyT3fl8SQ",
        "name": "CriticalPast",
        "publisher_type": "historical_footage_archive_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCLSRybRYNn8n6aK_R9X1M4A",
        "name": "CDO Canal del Deporte Olimpico",
        "publisher_type": "official_multisport_channel_targeted_search",
        "query": "esgrima",
    },
    {
        "id": "UCY2CGpvc5fMOHLOuDxy1OmA",
        "name": "Big State Sports",
        "publisher_type": "multisport_broadcaster_targeted_search",
        "query": "fencing",
    },
    {
        "id": "UCqEZpXu06_UEH1k4ENfMAQQ",
        "name": "InHigh TV",
        "publisher_type": "school_sports_broadcaster_targeted_search",
        "query": "フェンシング",
    },
    {
        "id": "UCWkqxFZQSdkR_q8kGz6p9mw",
        "name": "NCC Nagasaki",
        "publisher_type": "regional_broadcaster_targeted_search",
        "query": "フェンシング",
    },
)

YOUTUBE_ID_PATTERN = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|live/))"
    r"([A-Za-z0-9_-]{11})"
)


def _uploads_url(channel_id: str) -> str:
    if not channel_id.startswith("UC"):
        raise ValueError(f"unexpected YouTube channel ID: {channel_id}")
    return f"https://www.youtube.com/playlist?list=UU{channel_id[2:]}"


def _search_url(channel_id: str, query: str) -> str:
    return (
        f"https://www.youtube.com/channel/{channel_id}/search?query={quote_plus(query)}"
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and value >= 0 else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def records_from_payload(
    specification: dict[str, str],
    payload: dict[str, Any],
    *,
    discovered_at: datetime,
) -> list[VideoRecord]:
    """Convert one flat uploads-playlist snapshot into link-only records."""

    channel_id = specification["id"]
    observed_channel_id = str(payload.get("channel_id") or "")
    if observed_channel_id and observed_channel_id != channel_id:
        raise ValueError(
            f"YouTube channel identity changed for {channel_id}: {observed_channel_id}"
        )
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"YouTube channel {channel_id} has no entries list")

    channel_url = f"https://www.youtube.com/channel/{channel_id}"
    records: list[VideoRecord] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        identifier = str(raw.get("id") or "").strip()
        title = str(raw.get("title") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", identifier) or not title or identifier in seen:
            continue
        seen.add(identifier)
        source_url = f"https://www.youtube.com/watch?v={identifier}"
        candidates = infer_weapon_candidates(title)
        duration = _optional_float(raw.get("duration"))
        records.append(
            VideoRecord(
                video_id=f"youtube:{identifier}",
                title=title,
                weapon=primary_weapon(candidates),
                weapon_candidates=list(candidates),
                source_kind=SourceKind.YOUTUBE,
                source_url=HttpUrl(source_url),
                landing_page_url=HttpUrl(source_url),
                license_status=LicenseStatus.LINK_ONLY,
                duration_seconds=duration,
                discovered_at=discovered_at,
                reviewed=False,
                group_id=f"youtube:{identifier}",
                content_kind=classify_content(title, duration_seconds=duration),
                view_count=_optional_int(raw.get("view_count")),
                metadata={
                    "channel_id": channel_id,
                    "channel": specification["name"],
                    "channel_url": channel_url,
                    "publisher_type": specification["publisher_type"],
                    "catalog_method": specification.get(
                        "catalog_method", "yt_dlp_flat_uploads_playlist"
                    ),
                    "targeted_search_query": specification.get("query"),
                    "automatic_weapon_label": True,
                },
                notes=(
                    "metadata-only; media not downloaded; channel inclusion is not a rights grant; "
                    f"weapon_candidates={','.join(weapon.value for weapon in candidates) or 'none'}"
                ),
            )
        )
    records.sort(key=lambda record: record.video_id)
    return records


def _extract_youtube_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    if not path.exists():
        return identifiers
    for row in read_jsonl(path):
        video_id = str(row.get("video_id") or "")
        if video_id.startswith("youtube:") and len(video_id) == 19:
            identifiers.add(video_id.removeprefix("youtube:"))
        for key in ["source_url", "landing_page_url", "video_url", "url"]:
            match = YOUTUBE_ID_PATTERN.search(str(row.get(key) or ""))
            if match:
                identifiers.add(match.group(1))
    return identifiers


def fetch_public_catalog(
    cache_directory: Path,
    output_path: Path,
    report_path: Path,
    *,
    overlap_manifests: list[Path],
    force: bool = False,
) -> dict[str, Any]:
    """Enumerate public uploads without downloading or requesting playback media."""

    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - optional dependency guidance
        raise RuntimeError("Install pisteflow-analytics[collectors] for YouTube metadata") from exc

    cache_directory.mkdir(parents=True, exist_ok=True)
    discovered_at = datetime.now(UTC)
    all_records: list[VideoRecord] = []
    failures: list[dict[str, str]] = []
    raw_counts: dict[str, int] = {}
    catalogs = [
        {
            **specification,
            "catalog_id": specification["id"],
            "catalog_url": _uploads_url(specification["id"]),
            "cache_name": f"{specification['id']}.json",
        }
        for specification in CHANNELS
    ]
    catalogs.extend(
        {
            **specification,
            "catalog_id": f"{specification['id']}:search:{specification['query']}",
            "catalog_url": _search_url(specification["id"], specification["query"]),
            "cache_name": (
                f"{specification['id']}-search-"
                f"{re.sub(r'[^a-z0-9]+', '-', specification['query'].casefold()).strip('-')}.json"
            ),
            "catalog_method": "yt_dlp_flat_channel_search",
        }
        for specification in TARGETED_SEARCHES
    )
    for specification in catalogs:
        channel_id = specification["id"]
        catalog_id = specification["catalog_id"]
        cache_path = cache_directory / specification["cache_name"]
        if force:
            cache_path.unlink(missing_ok=True)
        try:
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                options = {
                    "extract_flat": "in_playlist",
                    "skip_download": True,
                    "quiet": True,
                    "no_warnings": True,
                    "ignoreerrors": True,
                    "socket_timeout": 30,
                    "retries": 3,
                }
                with yt_dlp.YoutubeDL(options) as downloader:
                    payload = downloader.extract_info(
                        specification["catalog_url"], download=False
                    )
                if not isinstance(payload, dict):
                    raise ValueError("empty uploads-playlist payload")
                write_json(cache_path, payload)
            entries = payload.get("entries", [])
            raw_counts[catalog_id] = len(entries) if isinstance(entries, list) else 0
            all_records.extend(
                records_from_payload(specification, payload, discovered_at=discovered_at)
            )
        except Exception as exc:  # yt-dlp exposes extractor-specific errors
            failures.append(
                {
                    "channel_id": channel_id,
                    "channel": specification["name"],
                    "catalog_id": catalog_id,
                    "error": str(exc),
                }
            )

    by_id: dict[str, VideoRecord] = {}
    duplicate_catalog_rows = 0
    for record in all_records:
        if record.video_id in by_id:
            duplicate_catalog_rows += 1
            continue
        by_id[record.video_id] = record
    records = sorted(by_id.values(), key=lambda record: record.video_id)
    write_jsonl(output_path, [record.model_dump(mode="json") for record in records])

    record_ids = {record.video_id.removeprefix("youtube:") for record in records}
    overlap_counts = {
        path.name: len(record_ids.intersection(_extract_youtube_ids(path)))
        for path in overlap_manifests
        if path.exists()
    }
    by_channel = Counter(str(record.metadata["channel"]) for record in records)
    summary: dict[str, Any] = {
        "retrieved_at": discovered_at.isoformat(),
        "fencing_focused_channels_configured": len(CHANNELS),
        "targeted_multisport_channel_searches_configured": len(TARGETED_SEARCHES),
        "catalog_endpoints_loaded": len(catalogs) - len(failures),
        "failures": failures,
        "unique_public_video_links": len(records),
        "raw_rows_by_channel_id": dict(sorted(raw_counts.items())),
        "records_by_channel": dict(sorted(by_channel.items())),
        "by_weapon": dict(
            sorted(Counter(record.weapon.value for record in records).items())
        ),
        "by_content_kind": dict(
            sorted(Counter(record.content_kind for record in records).items())
        ),
        "explicit_single_weapon_title_rows": sum(
            len(record.weapon_candidates) == 1 for record in records
        ),
        "mixed_weapon_title_rows": sum(
            len(record.weapon_candidates) > 1 for record in records
        ),
        "hours_with_duration_metadata": round(
            sum(record.duration_seconds or 0 for record in records) / 3_600,
            2,
        ),
        "duplicate_video_ids_within_catalog": duplicate_catalog_rows,
        "overlap_with_existing_manifests": dict(sorted(overlap_counts.items())),
        "claim_boundary": (
            f"Public viewing-link metadata from {len(CHANNELS)} sport-fencing-focused "
            "federation, community, coaching, analysis, bout-archive, and wheelchair-fencing "
            f"channels, plus {len(TARGETED_SEARCHES)} targeted searches within large "
            "multisport channels. No video bytes or "
            "playback URLs are acquired or redistributed. Counts are unique YouTube IDs, not "
            "content-hash-unique bouts, independent athletes, licensed training files, or reviewed "
            "labels; different channels may re-upload the same underlying footage. Weapon and "
            "content labels are title-derived discovery hints. Publisher-type metadata records "
            "the observed channel role and is not a media-rights grant."
        ),
        "output": str(output_path),
    }
    write_json(report_path, summary)
    return summary
