"""Command-line entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pisteflow.analysis.asmi_referee import train_and_report as train_asmi_referee
from pisteflow.analysis.competition_elo import train_and_report as train_competition_elo
from pisteflow.analysis.fencingly_diagnostic import train_and_report as train_fencingly_diagnostic
from pisteflow.analysis.hot_hand_validation import train_and_report as train_hot_hand_validation
from pisteflow.analysis.mfad_actions import train_and_report as train_mfad_actions
from pisteflow.analysis.nz_elo import train_and_report as train_nz_elo
from pisteflow.analysis.osf_pool_outcomes import train_and_report as train_osf_pool_outcomes
from pisteflow.analysis.pose_outcomes import train_and_report as train_pose_outcomes
from pisteflow.analysis.ranking_trajectory import train_and_report as train_ranking_trajectory
from pisteflow.analysis.sabre_eye_outcomes import train_and_report as train_sabre_eye_outcomes
from pisteflow.analysis.spatial_access import analyze_and_report as analyze_spatial_access
from pisteflow.analysis.strength_velocity import train_and_report as train_strength_velocity
from pisteflow.commands.showcase import app as showcase_app
from pisteflow.config import PATHS
from pisteflow.provenance import write_json
from pisteflow.sources import (
    activitynet,
    askfred_v1,
    asmi_referee,
    brazilian_ssbs,
    british_fencing,
    cmu_mocap,
    cmu_score_detection,
    commons,
    commons_images,
    efc_results,
    egohumans,
    epee_moves,
    facts,
    fencevision_blade,
    fencing_matches_ai,
    fencing_points,
    fencing_ref_metadata,
    fencing_vision,
    fencingly,
    fencingstatsnz,
    fencingtracker_community,
    fencingtv,
    foil_history,
    france_fencing_infrastructure,
    france_fencing_licences,
    frontiers_handedness_2021,
    g1_moves_fencing,
    github_fencing_media,
    hacs,
    handedness_rankings,
    harmony4d,
    hmdb51,
    hot_hand,
    hzume_epee_annotations,
    image_union_audit,
    injury_surveys,
    internet_archive,
    iot_balance_feedback,
    kaggle_fie_rankings,
    kinetics,
    korea_gkl_wheelchair,
    korean_fencing_tracker,
    lerobot_fencing,
    loc_fencing_images,
    log45,
    mfad,
    mit_ocw,
    mpii_pose,
    ncaa,
    olympics,
    oman,
    openimages_fencing,
    openverse_fencing,
    osf_epee_attention,
    osf_fencer_psychology,
    osf_sabre_eye_tracking,
    osf_three_weapons,
    paris_2024,
    peerj_strength,
    peerj_tdcs,
    personalized_strength,
    physiology,
    physionet_fencer_cardio,
    plos_elite_backpain,
    plos_fencer_cognition,
    plos_flick,
    plos_simulator,
    prod_ric_actions,
    recent_sabre,
    regional_tournament_ledgers,
    relative_age_combat,
    riposte_adapted_fencing,
    rt_fencing,
    saber_predict,
    scoping_review,
    sleep_wake,
    sports_images,
    spws_rankings,
    temporal_demands,
    temporal_demands_raw,
    tibial_impact,
    tokyo_epee,
    ucf101,
    us_population,
    usa_team_archives,
    usafencing,
    usafencing_clubs,
    wheelchair_foil_temporal_2025,
    wheelchair_oxygen,
    wikidata,
    wikidata_images,
    womens_foil_snapshot,
    womensports_fencing,
    youth_athletes,
    youtube,
    youtube8m_fencing,
    youtube_channels,
    zenodo_fencing_errors,
    zenodo_wheelchair_training,
)
from pisteflow.training.weapon import train_weapon_classifier
from pisteflow.video.features import VideoFeatureConfig, extract_manifest_features

app = typer.Typer(no_args_is_help=True, help="Three-weapon fencing analytics toolkit.")
sources_app = typer.Typer(no_args_is_help=True, help="Collect traceable public inputs.")
video_app = typer.Typer(no_args_is_help=True, help="Extract video features and train baselines.")
analysis_app = typer.Typer(no_args_is_help=True, help="Run leakage-aware fencing analyses.")
app.add_typer(sources_app, name="sources")
app.add_typer(video_app, name="video")
app.add_typer(analysis_app, name="analysis")
app.add_typer(showcase_app, name="showcase")
console = Console()


def _user_agent() -> str:
    return os.environ.get("PISTEFLOW_USER_AGENT", commons.DEFAULT_USER_AGENT)


@sources_app.command("commons-discover")
def commons_discover(
    curated: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.repository / "data/registry/commons_curated.yml",
    output: Annotated[Path, typer.Option()] = (
        PATHS.repository / "data/manifests/commons-videos.jsonl"
    ),
) -> None:
    records = commons.discover_curated(curated, output, user_agent=_user_agent())
    summary = commons.summarize(records)
    write_json(PATHS.repository / "data/manifests/commons-videos-summary.json", summary)
    console.print_json(json.dumps(summary))


@sources_app.command("commons-download")
def commons_download(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.repository / "data/manifests/commons-videos.jsonl",
    max_gib: Annotated[float | None, typer.Option(min=0.01)] = None,
    workers: Annotated[int, typer.Option(min=1, max=6)] = 4,
) -> None:
    PATHS.create_local_directories()
    limit = int(max_gib * 1024**3) if max_gib is not None else None
    records, failures = commons.download_open_media(
        manifest,
        PATHS.raw / "commons",
        max_total_bytes=limit,
        user_agent=_user_agent(),
        report_path=PATHS.cache / "commons-download-report.json",
        workers=workers,
    )
    summary = commons.summarize(records)
    summary["failures"] = failures
    summary["claim_boundary"] = (
        f"{len(records)} manually curated epee, foil, and sabre Commons videos with per-file open "
        "licenses. Weapon labels are curator-reviewed source labels, not frame-level actions. "
        "Evaluation splits also isolate named source families, but "
        f"{len(records)} videos remain a small internal baseline rather than in-the-wild "
        "deployment evidence."
    )
    write_json(PATHS.repository / "data/manifests/commons-videos-summary.json", summary)
    console.print_json(json.dumps(summary))


@sources_app.command("commons-images-discover")
def commons_images_discover(
    max_depth: Annotated[int, typer.Option(min=0, max=3)] = 2,
    refresh: bool = False,
) -> None:
    """Discover modern-sport still images from controlled three-weapon category trees."""

    PATHS.create_local_directories()
    summary = commons_images.discover(
        PATHS.repository / "data/manifests/commons-fencing-images.jsonl",
        PATHS.repository / "data/manifests/commons-fencing-images-summary.json",
        PATHS.cache / "commons-image-categories",
        user_agent=_user_agent(),
        max_depth=max_depth,
        refresh=refresh,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("commons-images-download")
def commons_images_download(
    workers: Annotated[int, typer.Option(min=1, max=2)] = 1,
    thumbnail_width: Annotated[int, typer.Option(min=320, max=2560)] = 1280,
) -> None:
    """Download license-verified Commons stills with checksums and attribution."""

    PATHS.create_local_directories()
    summary = commons_images.download(
        PATHS.repository / "data/manifests/commons-fencing-images.jsonl",
        PATHS.raw / "commons-images",
        PATHS.repository / "data/manifests/commons-fencing-images-download-summary.json",
        user_agent=_user_agent(),
        workers=workers,
        thumbnail_width=thumbnail_width,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("image-union-audit")
def image_union_audit_command() -> None:
    """Audit exact and upstream-identity overlap across all checksummed still-image sources."""

    summary = image_union_audit.audit(
        PATHS.repository,
        PATHS.repository / "data/manifests/image-union-audit.jsonl",
        PATHS.repository / "data/manifests/image-union-audit-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fie-youtube-import")
def fie_youtube_import(
    input_path: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.cache / "fie-youtube-channel.json",
    output: Annotated[Path, typer.Option()] = (
        PATHS.repository / "data/manifests/fie-youtube-videos.jsonl"
    ),
) -> None:
    records = youtube.import_flat_playlist(input_path, output)
    summary = youtube.summarize(records)
    summary["output"] = str(output)
    summary["claim_boundary"] = (
        "Official FIE YouTube metadata and public viewing links only. Media are not downloaded "
        "or redistributed; weapon and content labels are title-derived discovery aids, not "
        "reviewed ground truth."
    )
    write_json(PATHS.repository / "data/manifests/fie-youtube-videos-summary.json", summary)
    console.print_json(json.dumps(summary))


@sources_app.command("youtube-channels-fetch")
def youtube_channels_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh every configured public uploads-playlist snapshot."),
    ] = False,
) -> None:
    """Enumerate configured fencing channels and targeted searches as public links."""

    PATHS.create_local_directories()
    manifest_directory = PATHS.repository / "data/manifests"
    summary = youtube_channels.fetch_public_catalog(
        PATHS.cache / "youtube-fencing-channels",
        manifest_directory / "youtube-fencing-channels.jsonl",
        manifest_directory / "youtube-fencing-channels-summary.json",
        overlap_manifests=[
            manifest_directory / "fie-youtube-videos.jsonl",
            manifest_directory / "youtube8m-fencing.jsonl",
            manifest_directory / "fencing-reference-video-links.jsonl",
            manifest_directory / "saber-predict-video-links.jsonl",
            manifest_directory / "activitynet-fencing.jsonl",
            manifest_directory / "hacs-fencing-clips.jsonl",
            manifest_directory / "kinetics-fencing.jsonl",
        ],
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("hot-hand-fetch")
def hot_hand_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned source archive.")] = False,
) -> None:
    """Prepare the mirrored-row-audited February 2026 NAC foil snapshot."""

    PATHS.create_local_directories()
    summary = hot_hand.fetch_and_prepare(
        PATHS.cache / "source-archives/fencing-hot-hand-9e748e7.tar.gz",
        PATHS.processed / "february-nac-2026-hot-hand-bouts.parquet",
        PATHS.repository / "data/manifests/february-nac-2026-hot-hand-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("asmi-referee-fetch")
def asmi_referee_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned source archive.")] = False,
) -> None:
    """Verify the 395 local clips and extract filename-grouped visual features."""

    PATHS.create_local_directories()
    summary = asmi_referee.fetch_and_prepare(
        PATHS.cache / "source-archives/fencing-ai-referee-ef9ace1.tar.gz",
        PATHS.cache / "fencing-ai-referee",
        PATHS.processed / "asmi-referee-video-features.parquet",
        PATHS.repository / "data/manifests/asmi-referee-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("korean-fencing-tracker-fetch")
def korean_fencing_tracker_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned source archive.")] = False,
) -> None:
    """Prepare three-weapon OCR events and isolated unreviewed weak labels."""

    PATHS.create_local_directories()
    summary = korean_fencing_tracker.fetch_and_prepare(
        PATHS.cache / "source-archives/korean-fencing-tracker-3a40997.tar.gz",
        PATHS.cache / "korean-fencing-tracker",
        PATHS.repository / "data/manifests/korean-fencing-tracker-videos.jsonl",
        PATHS.repository / "data/manifests/korean-fencing-tracker-reports.jsonl",
        PATHS.processed / "korean-fencing-tracker-touches.parquet",
        PATHS.processed / "korean-fencing-tracker-phrases.parquet",
        PATHS.processed / "korean-fencing-tracker-action-proxies.parquet",
        PATHS.processed / "korean-fencing-tracker-priority.parquet",
        PATHS.repository / "data/manifests/korean-fencing-tracker-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("hzume-epee-annotations-fetch")
def hzume_epee_annotations_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned source archive.")] = False,
) -> None:
    """Prepare the local-only 22-video epee temporal-annotation inventory."""

    PATHS.create_local_directories()
    summary = hzume_epee_annotations.fetch_and_prepare(
        PATHS.cache / "source-archives/hzume-epee-fb83a227.tar.gz",
        PATHS.cache / "hzume-epee-action-recognition",
        PATHS.processed / "hzume-epee-action-annotations.parquet",
        PATHS.repository / "data/manifests/hzume-epee-action-annotations-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("prod-ric-actions-fetch")
def prod_ric_actions_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned seven-video archive.")] = False,
) -> None:
    """Audit source-grouped demonstration and epee broadcast clips."""

    PATHS.create_local_directories()
    summary = prod_ric_actions.fetch_and_prepare(
        PATHS.cache / "source-archives/prod-ric-fencing-5da4420.tar.gz",
        PATHS.cache / "prod-ric-fencing-action",
        PATHS.processed / "prod-ric-fencing-video-inventory.parquet",
        PATHS.repository / "data/manifests/prod-ric-fencing-video-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("github-fencing-media-fetch")
def github_fencing_media_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the three pinned local-only GitHub media snapshots."),
    ] = False,
) -> None:
    """Audit ten original three-weapon videos plus grouped still derivatives."""

    PATHS.create_local_directories()
    summary = github_fencing_media.fetch_and_prepare(
        PATHS.cache,
        PATHS.processed / "github-fencing-media-inventory.parquet",
        PATHS.repository / "data/manifests/github-fencing-media-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("g1-moves-fencing-fetch")
def g1_moves_fencing_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the pinned CC BY 4.0 two-motion subset."),
    ] = False,
) -> None:
    """Acquire two generic fencing motion-capture clips for local pretraining."""

    PATHS.create_local_directories()
    summary = g1_moves_fencing.fetch_and_prepare(
        PATHS.cache / "g1-moves-fencing",
        PATHS.processed / "g1-moves-fencing-motion.parquet",
        PATHS.repository / "data/manifests/g1-moves-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("osf-three-weapons-fetch")
def osf_three_weapons_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the pinned CC BY 4.0 OSF source archive."),
    ] = False,
) -> None:
    """Prepare 158,652 privacy-minimized three-weapon junior pool bouts."""

    PATHS.create_local_directories()
    summary = osf_three_weapons.fetch_and_prepare(
        PATHS.cache / "source-archives/three-weapons-osf-deposit.zip",
        PATHS.processed / "osf-three-weapons",
        PATHS.repository / "data/manifests/osf-three-weapons-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("osf-sabre-eye-tracking-fetch")
def osf_sabre_eye_tracking_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the pinned public but unlicensed OSF deposits."),
    ] = False,
) -> None:
    """Prepare local-only main and pilot sabre eye-tracking tables."""

    PATHS.create_local_directories()
    summary = osf_sabre_eye_tracking.fetch_and_prepare(
        PATHS.cache / "source-archives/osf-sabre-eye-tracking",
        PATHS.processed / "osf-sabre-eye-tracking",
        PATHS.repository / "data/manifests/osf-sabre-eye-tracking-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("osf-fencer-psychology-fetch")
def osf_fencer_psychology_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the pinned public but unlicensed OSF SPSS file."),
    ] = False,
) -> None:
    """Prepare a local-only privacy-minimized 78-fencer survey table."""

    PATHS.create_local_directories()
    summary = osf_fencer_psychology.fetch_and_prepare(
        PATHS.cache / "source-archives/osf-fencer-psychology.sav",
        PATHS.processed / "osf-fencer-psychology.parquet",
        PATHS.repository / "data/manifests/osf-fencer-psychology-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("osf-epee-attention-fetch")
def osf_epee_attention_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the pinned view-only, unlicensed OSF files."),
    ] = False,
) -> None:
    """Prepare local-only attack and visual-attention tables for 24 epee fencers."""

    PATHS.create_local_directories()
    summary = osf_epee_attention.fetch_and_prepare(
        PATHS.cache / "source-archives/osf-epee-attention",
        PATHS.processed / "osf-epee-attack-participants.parquet",
        PATHS.processed / "osf-epee-covat-conditions.parquet",
        PATHS.repository / "data/manifests/osf-epee-attention-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("iot-balance-feedback-fetch")
def iot_balance_feedback_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the pinned CC BY Europe PMC supplement."),
    ] = False,
) -> None:
    """Prepare 30 juvenile fencers' Week 0 and Week 5 balance measurements."""

    PATHS.create_local_directories()
    summary = iot_balance_feedback.fetch_and_prepare(
        PATHS.cache / "source-archives/iot-balance-feedback",
        PATHS.processed / "iot-balance-feedback-participant-weeks.parquet",
        PATHS.repository / "data/manifests/iot-balance-feedback-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("plos-fencer-cognition-fetch")
def plos_fencer_cognition_fetch(
    force: Annotated[bool, typer.Option(help="Refresh both pinned CC BY workbooks.")] = False,
) -> None:
    """Prepare the 18-fencer cognitive, fitness, and blood-measure panel."""

    PATHS.create_local_directories()
    summary = plos_fencer_cognition.fetch_and_prepare(
        PATHS.cache / "source-archives/plos-fencer-cognition",
        PATHS.processed / "plos-fencer-cognition.parquet",
        PATHS.repository / "data/manifests/plos-fencer-cognition-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("plos-elite-backpain-fetch")
def plos_elite_backpain_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned CC BY SPSS file.")] = False,
) -> None:
    """Prepare the privacy-minimized 23-fencer elite back-pain subset."""

    PATHS.create_local_directories()
    summary = plos_elite_backpain.fetch_and_prepare(
        PATHS.cache / "source-archives/plos-elite-backpain.sav",
        PATHS.processed / "plos-elite-backpain.parquet",
        PATHS.repository / "data/manifests/plos-elite-backpain-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("spws-rankings-fetch")
def spws_rankings_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned source archive.")] = False,
) -> None:
    """Prepare privacy-minimized three-weapon veterans ranking and event panels."""

    PATHS.create_local_directories()
    summary = spws_rankings.fetch_and_prepare(
        PATHS.cache / "source-archives/spws-rankings-b0631c4.tar.gz",
        PATHS.cache / "spws-rankings",
        PATHS.repository / "data/manifests/spws-ranking-workbooks.jsonl",
        PATHS.processed / "spws-ranking-memberships.parquet",
        PATHS.processed / "spws-event-results.parquet",
        PATHS.repository / "data/manifests/spws-rankings-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("openimages-fencing-prepare")
def openimages_fencing_prepare() -> None:
    """Filter all six human-verified fencing-related Open Images classes."""

    PATHS.create_local_directories()
    summary = openimages_fencing.prepare(
        PATHS.cache / "openimages-v7",
        PATHS.repository / "data/manifests/openimages-fencing-images.jsonl",
        PATHS.repository / "data/manifests/openimages-fencing-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("openimages-fencing-download")
def openimages_fencing_download(
    workers: Annotated[int, typer.Option(min=1, max=12)] = 6,
) -> None:
    """Download all positive target images locally from the official S3 mirror."""

    PATHS.create_local_directories()
    summary = openimages_fencing.download_positive_images(
        PATHS.repository / "data/manifests/openimages-fencing-images.jsonl",
        PATHS.raw / "openimages-v7",
        PATHS.repository / "data/manifests/openimages-fencing-download-summary.json",
        user_agent=_user_agent(),
        workers=workers,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("openimages-fencing-enrich-metadata")
def openimages_fencing_enrich_metadata() -> None:
    """Join official per-image license, attribution, and original-source metadata."""

    PATHS.create_local_directories()
    summary = openimages_fencing.enrich_image_metadata(
        PATHS.repository / "data/manifests/openimages-fencing-images.jsonl",
        PATHS.cache / "openimages-v7",
        PATHS.repository / "data/manifests/openimages-fencing-metadata-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("zenodo-wheelchair-training-prepare")
def zenodo_wheelchair_training_prepare() -> None:
    """Prepare pinned group aggregates from two 2026 wheelchair-fencing studies."""

    PATHS.create_local_directories()
    summary = zenodo_wheelchair_training.prepare(
        PATHS.cache / "zenodo-wheelchair-fencing",
        PATHS.processed / "zenodo-wheelchair-training-aggregates.parquet",
        PATHS.repository / "data/manifests/zenodo-wheelchair-training-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("kaggle-fie-rankings-prepare")
def kaggle_fie_rankings_prepare() -> None:
    """Prepare a local-only, pseudonymous 2021-22 global ranking panel."""

    PATHS.create_local_directories()
    root = PATHS.cache / "kaggle-discovery"
    extracted = root / "extracted/sethlogan__fie-fencer-rankings-end-of-20212022-season"
    summary = kaggle_fie_rankings.prepare(
        root / "sethlogan__fie-fencer-rankings-end-of-20212022-season.zip",
        extracted / "FIE Fencer Ranking (End of 2021-22 Season).xlsx",
        PATHS.processed / "kaggle-fie-2021-22-rankings.parquet",
        PATHS.processed / "kaggle-fie-2021-22-event-points.parquet",
        PATHS.repository / "data/manifests/kaggle-fie-2021-22-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("regional-tournament-ledgers-prepare")
def regional_tournament_ledgers_prepare() -> None:
    """Prepare restricted regional tournament ledgers as local pseudonymous rows."""

    PATHS.create_local_directories()
    repository = PATHS.cache / "github-discovery/aidanKnox__Fencing-Data-Analytics-Project"
    summary = regional_tournament_ledgers.prepare(
        repository,
        PATHS.processed / "regional-tournament-bouts.parquet",
        PATHS.processed / "regional-tournament-rosters.parquet",
        PATHS.repository / "data/manifests/regional-tournament-ledgers-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("usafencing-clubs-prepare")
def usafencing_clubs_prepare() -> None:
    """Prepare the CC0, direct-name-free USA Fencing club geography table."""

    PATHS.create_local_directories()
    summary = usafencing_clubs.prepare(
        PATHS.cache / "source-archives/usafencing-clubs-v1.xlsx",
        PATHS.cache / "source-archives/usafencing-clubs-v1-readme.md",
        PATHS.processed / "usafencing-clubs-geography.parquet",
        PATHS.repository / "data/manifests/usafencing-clubs-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("france-fencing-licences-fetch")
def france_fencing_licences_fetch(
    start_year: Annotated[int, typer.Option(min=2011, max=2023)] = 2011,
    end_year: Annotated[int, typer.Option(min=2011, max=2023)] = 2023,
    keep_source_archives: Annotated[
        bool,
        typer.Option(help="Retain the roughly 2.6 GB complete multi-sport source files."),
    ] = False,
    force: Annotated[bool, typer.Option(help="Refresh any retained annual source files.")] = False,
) -> None:
    """Prepare France's open, geocoded annual fencing-licence census."""

    PATHS.create_local_directories()
    summary = france_fencing_licences.fetch_and_prepare(
        PATHS.cache / "france-sports-licences",
        PATHS.raw / "france-fencing-licences",
        PATHS.processed / "france-fencing-licences.parquet",
        PATHS.processed / "france-fencing-licence-history-2002-2024.parquet",
        PATHS.processed / "france-fencing-licences-2024-department-age-sex.parquet",
        PATHS.repository / "data/manifests/france-fencing-licences-summary.json",
        user_agent=_user_agent(),
        start_year=start_year,
        end_year=end_year,
        keep_source_archives=keep_source_archives,
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("france-fencing-infrastructure-fetch")
def france_fencing_infrastructure_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned portal exports.")] = False,
) -> None:
    """Prepare France's open fencing-club and facility geography tables."""

    PATHS.create_local_directories()
    summary = france_fencing_infrastructure.fetch_and_prepare(
        PATHS.cache / "france-fencing-infrastructure",
        PATHS.processed / "france-fencing-clubs-2022-2024.parquet",
        PATHS.processed / "france-fencing-equipment.parquet",
        PATHS.repository / "data/manifests/france-fencing-infrastructure-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencingly-fetch")
def fencingly_fetch(
    workers: Annotated[int, typer.Option(min=1, max=8)] = 4,
) -> None:
    """Audit and exact-deduplicate both large FENCINGLY pose archives."""

    PATHS.create_local_directories()
    summary = fencingly.fetch_and_prepare(
        PATHS.cache / "fencingly",
        PATHS.cache / "fencingly/extracted",
        PATHS.processed / "fencingly-pose-features.parquet",
        PATHS.repository / "data/manifests/fencingly-summary.json",
        user_agent=_user_agent(),
        workers=workers,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fie-playlists-fetch")
def fie_playlists_fetch(
    index_path: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.cache / "fie-youtube-playlists.json",
    output: Annotated[Path, typer.Option()] = (
        PATHS.repository / "data/manifests/fie-youtube-playlist-members.jsonl"
    ),
    workers: Annotated[int, typer.Option(min=1, max=4)] = 2,
) -> None:
    PATHS.create_local_directories()
    summary = youtube.fetch_playlist_members(
        index_path,
        PATHS.cache / "fie-youtube-playlists",
        output,
        workers=workers,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fie-youtube-enrich")
def fie_youtube_enrich(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.repository / "data/manifests/fie-youtube-videos.jsonl",
    memberships: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.repository / "data/manifests/fie-youtube-playlist-members.jsonl",
) -> None:
    records = youtube.enrich_from_playlists(manifest, memberships)
    summary = youtube.summarize(records)
    summary["output"] = str(manifest)
    console.print_json(json.dumps(summary))


@sources_app.command("fencingtv-fetch")
def fencingtv_fetch(
    output: Annotated[Path, typer.Option()] = (
        PATHS.repository / "data/manifests/fencingtv-videos.jsonl"
    ),
) -> None:
    records = fencingtv.fetch_public_catalog(output, user_agent=_user_agent())
    summary = fencingtv.summarize(records)
    summary["output"] = str(output)
    summary["claim_boundary"] = (
        "Public FencingTV catalog metadata and landing-page links only. Playback media are not "
        "requested or redistributed, and automated weapon labels are not training ground truth."
    )
    write_json(PATHS.repository / "data/manifests/fencingtv-videos-summary.json", summary)
    console.print_json(json.dumps(summary))


@sources_app.command("efc-results-fetch")
def efc_results_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the official event catalog pages.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = efc_results.fetch_and_prepare(
        PATHS.cache / "efc-results",
        PATHS.repository / "data/manifests/efc-results.jsonl",
        PATHS.repository / "data/manifests/efc-results-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("harmony4d-fetch")
def harmony4d_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the official remote inventory.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = harmony4d.fetch_and_prepare(
        PATHS.cache / "harmony4d-tree.json",
        PATHS.repository / "data/manifests/harmony4d-fencing-archives.jsonl",
        PATHS.repository / "data/manifests/harmony4d-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    catalog_summary = harmony4d.prepare_voxel51_catalog(
        PATHS.cache / "harmony4d/samples.json",
        PATHS.cache / "harmony4d/metadata.json",
        PATHS.cache / "harmony4d/sword-preview/cam01-20.mp4",
        PATHS.repository / "data/manifests/harmony4d-sword-samples.jsonl",
        PATHS.repository / "data/manifests/harmony4d-sword-catalog-summary.json",
    )
    console.print_json(json.dumps({"archives": summary, "catalog": catalog_summary}))


@sources_app.command("cmu-mocap-swordplay-prepare")
def cmu_mocap_swordplay_prepare() -> None:
    PATHS.create_local_directories()
    summary = cmu_mocap.prepare(
        PATHS.raw / "cmu-mocap-swordplay",
        PATHS.repository / "data/manifests/cmu-mocap-swordplay.jsonl",
        PATHS.repository / "data/manifests/cmu-mocap-swordplay-summary.json",
        user_agent=_user_agent(),
    )
    console.print_json(json.dumps(summary))


@sources_app.command("olympics-fetch")
def olympics_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the licensed raw snapshot.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = olympics.fetch_and_prepare(
        PATHS.cache / "olympic_athletes.rda",
        PATHS.processed / "olympic-fencing-athletes.parquet",
        PATHS.repository / "data/manifests/olympic-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("paris-2024-prepare")
def paris_2024_prepare(
    pdf_path: Annotated[
        Path,
        typer.Option(exists=True, readable=True, help="Official Paris 2024 fencing results book."),
    ] = PATHS.cache / "paris-2024-fencing-results-book.pdf",
) -> None:
    PATHS.create_local_directories()
    summary = paris_2024.prepare_pdf(
        pdf_path,
        PATHS.processed / "paris-2024-individual-bouts.parquet",
        PATHS.processed / "paris-2024-team-matches.parquet",
        PATHS.processed / "paris-2024-team-relays.parquet",
        PATHS.repository / "data/manifests/paris-2024-results-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("wikidata-fetch")
def wikidata_fetch(
    output: Annotated[Path, typer.Option()] = (
        PATHS.repository / "data/manifests/wikidata-fencers.jsonl"
    ),
    workers: Annotated[int, typer.Option(min=1, max=6)] = 4,
    force: Annotated[bool, typer.Option(help="Refresh every Wikidata snapshot.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = wikidata.build_manifest(
        PATHS.cache / "wikidata-fencing-human-ids-v2.json",
        PATHS.cache / "wikidata",
        output,
        PATHS.repository / "data/manifests/wikidata-fencers-summary.json",
        user_agent=_user_agent(),
        workers=workers,
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("openverse-fencing-discover")
def openverse_fencing_discover(
    refresh: Annotated[bool, typer.Option(help="Refresh cached Openverse API pages.")] = False,
) -> None:
    """Discover non-Wikimedia open-license fencing still-image candidates."""

    PATHS.create_local_directories()
    summary = openverse_fencing.discover(
        PATHS.repository / "data/manifests/openverse-fencing-images.jsonl",
        PATHS.repository / "data/manifests/openverse-fencing-summary.json",
        PATHS.cache / "openverse-fencing",
        user_agent=_user_agent(),
        refresh=refresh,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("openverse-fencing-download")
def openverse_fencing_download() -> None:
    """Download bounded previews for the Openverse fencing candidate manifest."""

    PATHS.create_local_directories()
    summary = openverse_fencing.download(
        PATHS.repository / "data/manifests/openverse-fencing-images.jsonl",
        PATHS.raw / "openverse-fencing",
        PATHS.repository / "data/manifests/openverse-fencing-summary.json",
        user_agent=_user_agent(),
    )
    console.print_json(json.dumps(summary))


@sources_app.command("wikidata-images-discover")
def wikidata_images_discover() -> None:
    """Resolve Commons images attached to the Wikidata fencer registry."""

    summary = wikidata_images.discover(
        PATHS.repository / "data/manifests/wikidata-fencers.jsonl",
        PATHS.repository / "data/manifests/wikidata-fencer-images.jsonl",
        PATHS.repository / "data/manifests/wikidata-fencer-images-summary.json",
        user_agent=_user_agent(),
    )
    console.print_json(json.dumps(summary))


@sources_app.command("wikidata-images-download")
def wikidata_images_download(
    thumbnail_width: Annotated[int, typer.Option(min=320, max=2560)] = 1024,
    workers: Annotated[int, typer.Option(min=1, max=12)] = 6,
    network: Annotated[
        bool,
        typer.Option("--network/--reuse-only", help="Download new files or only reuse local ones."),
    ] = True,
) -> None:
    """Download open-license bounded thumbnails from the athlete-image manifest."""

    PATHS.create_local_directories()
    summary = wikidata_images.download(
        PATHS.repository / "data/manifests/wikidata-fencer-images.jsonl",
        PATHS.raw / "wikidata-fencer-images",
        PATHS.repository / "data/manifests/wikidata-fencer-images-summary.json",
        reuse_manifest_path=(
            PATHS.repository / "data/manifests/commons-fencing-images.jsonl"
        ),
        network=network,
        user_agent=_user_agent(),
        thumbnail_width=thumbnail_width,
        workers=workers,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("womensports-fencing-fetch")
def womensports_fencing_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned Mendeley archive.")] = False,
) -> None:
    """Extract and audit the broad Fencing class from WomenSports."""

    PATHS.create_local_directories()
    summary = womensports_fencing.fetch_and_prepare(
        PATHS.cache / "mendeley/womensports/m39ymdv48s-v3.zip",
        PATHS.cache / "mendeley/womensports/WSPORTS2024.zip",
        PATHS.raw / "womensports/fencing",
        PATHS.repository / "data/manifests/womensports-fencing-images.jsonl",
        PATHS.repository / "data/manifests/womensports-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("physionet-fencer-cardio-fetch")
def physionet_fencer_cardio_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned PhysioNet files.")] = False,
) -> None:
    """Prepare participant-grouped cardiorespiratory rows for ten teenage fencers."""

    PATHS.create_local_directories()
    summary = physionet_fencer_cardio.fetch_and_prepare(
        PATHS.cache / "physionet/actes-cycloergometer-1.0.0",
        PATHS.processed / "physionet-actes-fencer-profiles.parquet",
        PATHS.processed / "physionet-actes-fencer-cardio.parquet",
        PATHS.repository / "data/manifests/physionet-actes-fencer-cardio-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("wheelchair-foil-temporal-2025-fetch")
def wheelchair_foil_temporal_2025_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned Figshare SAV.")] = False,
) -> None:
    """Compatibility alias for the canonical wheelchair-foil temporal collector."""

    PATHS.create_local_directories()
    summary = wheelchair_foil_temporal_2025.fetch_and_prepare(
        PATHS.cache / "figshare/wheelchair-foil-temporal/wheelchair-foil-temporal.sav",
        PATHS.processed / "wheelchair-foil-temporal.parquet",
        PATHS.repository / "data/manifests/wheelchair-foil-temporal-codebook.json",
        PATHS.repository / "data/manifests/wheelchair-foil-temporal-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("ncaa-fetch")
def ncaa_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the public HTML snapshots.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = ncaa.fetch_and_prepare(
        PATHS.cache / "ncaa-results",
        PATHS.processed / "ncaa-athlete-seasons.parquet",
        PATHS.processed / "ncaa-pool-bouts.parquet",
        PATHS.repository / "data/manifests/ncaa-results-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("relative-age-combat-fetch")
def relative_age_combat_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned Figshare CSV.")] = False,
) -> None:
    """Prepare privacy-minimized relative-age rows for 405 youth fencers."""

    PATHS.create_local_directories()
    summary = relative_age_combat.fetch_and_prepare(
        PATHS.cache / "figshare/relative-age-combat/relative-age-combat-v2.csv",
        PATHS.processed / "relative-age-youth-fencers.parquet",
        PATHS.repository / "data/manifests/relative-age-youth-fencers-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("korea-gkl-wheelchair-fetch")
def korea_gkl_wheelchair_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned Korean public-data CSV.")] = False,
) -> None:
    """Prepare official Korean wheelchair-fencing events and name-free placements."""

    PATHS.create_local_directories()
    summary = korea_gkl_wheelchair.fetch_and_prepare(
        PATHS.cache / "data-go-kr/korea-gkl-wheelchair-fencing-20260322.csv",
        PATHS.processed / "korea-gkl-wheelchair-fencing-events.parquet",
        PATHS.processed / "korea-gkl-wheelchair-fencing-results.parquet",
        PATHS.repository / "data/manifests/korea-gkl-wheelchair-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("brazilian-ssbs-fencing-fetch")
def brazilian_ssbs_fencing_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned Science Data Bank XLSX.")] = False,
) -> None:
    """Prepare the local-only five-row fencing subset from the Brazilian SSBS study."""

    PATHS.create_local_directories()
    summary = brazilian_ssbs.fetch_and_prepare(
        PATHS.cache / "sciencedb/brazilian-ssbs/SBSS_final.xlsx",
        PATHS.processed / "brazilian-ssbs-fencers-local.parquet",
        PATHS.repository / "data/manifests/brazilian-ssbs-fencers-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("riposte-adapted-fencing-fetch")
def riposte_adapted_fencing_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned Figshare DOCX.")] = False,
) -> None:
    """Prepare adapted-fencing attendance and aggregate pilot outcomes."""

    PATHS.create_local_directories()
    summary = riposte_adapted_fencing.fetch_and_prepare(
        PATHS.cache / "figshare/riposte-adapted-fencing/riposte-datasheet.docx",
        PATHS.processed / "riposte-adapted-fencing-attendance.parquet",
        PATHS.processed / "riposte-adapted-fencing-aggregate-outcomes.parquet",
        PATHS.repository / "data/manifests/riposte-adapted-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("usafencing-fetch")
def usafencing_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the licensed Kaggle archive.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = usafencing.fetch_and_prepare(
        PATHS.cache / "kaggle" / "usa-fencing-national.zip",
        PATHS.processed / "usafencing-bouts.parquet",
        PATHS.processed / "usafencing-events.parquet",
        PATHS.repository / "data/manifests/usafencing-results-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("foil-history-fetch")
def foil_history_fetch(
    force: Annotated[
        bool, typer.Option(help="Refresh the pinned GPL repository snapshot.")
    ] = False,
) -> None:
    PATHS.create_local_directories()
    summary = foil_history.fetch_and_prepare(
        PATHS.cache / "fie-foil-history.zip",
        PATHS.processed / "fie-foil-history-bouts.parquet",
        PATHS.repository / "data/manifests/fie-foil-history-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencing-points-fetch")
def fencing_points_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the MIT-labelled Kaggle snapshot.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = fencing_points.fetch_and_prepare(
        PATHS.cache / "kaggle" / "fencing-points-mf.zip",
        PATHS.processed / "fencing-points-mf.parquet",
        PATHS.repository / "data/manifests/fencing-points-mf-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("british-fencing-fetch")
def british_fencing_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the nine official public tables.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = british_fencing.fetch_and_prepare(
        PATHS.cache / "british-fencing",
        PATHS.processed / "british-fencing-performance.parquet",
        PATHS.repository / "data/manifests/british-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("rt-fencing-prepare")
def rt_fencing_prepare(
    force: Annotated[bool, typer.Option(help="Refresh the public Google Drive archive.")] = False,
) -> None:
    PATHS.create_local_directories()
    archive = rt_fencing.acquire(PATHS.cache / "rt-fencing" / "dataRGB.zip", force=force)
    summary = rt_fencing.prepare(
        archive,
        PATHS.repository / "data/manifests/rt-fencing-images.jsonl",
        PATHS.repository / "data/manifests/rt-fencing-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("facts-fetch")
def facts_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the public FACTS annotations.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = facts.fetch_and_prepare(
        PATHS.cache / "facts" / "filtered_data_800_fencing.csv",
        PATHS.processed / "facts-fencing-actions.parquet",
        PATHS.repository / "data/manifests/facts-fencing-actions-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencing-matches-ai-fetch")
def fencing_matches_ai_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the local-only Kaggle archive.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = fencing_matches_ai.fetch_and_prepare(
        PATHS.cache / "kaggle" / "sport-fencing-matches-ai.zip",
        PATHS.processed / "fencing-matches-ai-pose-features.parquet",
        PATHS.repository / "data/manifests/fencing-matches-ai-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("usa-team-archives-fetch")
def usa_team_archives_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the local-only ranking archive.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = usa_team_archives.fetch_and_prepare(
        PATHS.cache / "kaggle" / "extra" / "usa-team-archives.zip",
        PATHS.processed / "usa-team-ranking-history.parquet",
        PATHS.repository / "data/manifests/usa-team-ranking-history-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("womens-foil-snapshot-fetch")
def womens_foil_snapshot_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the historical Kaggle snapshot.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = womens_foil_snapshot.fetch_and_prepare(
        PATHS.cache / "kaggle" / "fie-womens-foil.zip",
        PATHS.processed / "fie-womens-foil-bios.parquet",
        PATHS.processed / "fie-womens-foil-rankings.parquet",
        PATHS.processed / "fie-womens-foil-tournaments.parquet",
        PATHS.repository / "data/manifests/fie-womens-foil-snapshot-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("zenodo-fencing-errors-prepare")
def zenodo_fencing_errors_prepare(
    force_tails: Annotated[
        bool, typer.Option(help="Refresh the small remote ZIP inventory tails.")
    ] = False,
) -> None:
    PATHS.create_local_directories()
    summary = zenodo_fencing_errors.prepare(
        PATHS.cache / "zenodo" / "fencing-errors",
        PATHS.cache / "zenodo" / "fencing-errors" / "inventory",
        PATHS.repository / "data/manifests/zenodo-fencing-errors.jsonl",
        PATHS.repository / "data/manifests/zenodo-fencing-errors-summary.json",
        user_agent=_user_agent(),
        force_tails=force_tails,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("cmu-score-detection-fetch")
def cmu_score_detection_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the three CMU score datasets.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = cmu_score_detection.fetch_and_prepare(
        PATHS.cache / "huggingface" / "cmu-score-detection",
        PATHS.repository / "data/manifests/cmu-touch-images.jsonl",
        PATHS.processed / "cmu-score-signals.parquet",
        PATHS.processed / "cmu-scoreboard-yolo-metadata.parquet",
        PATHS.repository / "data/manifests/cmu-score-detection-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("tokyo-epee-fetch")
def tokyo_epee_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned MIT workbook.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = tokyo_epee.fetch_and_prepare(
        PATHS.cache / "community" / "tokyo-epee" / "FencingData.xlsx",
        PATHS.processed / "tokyo-2020-epee-touch-events.parquet",
        PATHS.repository / "data/manifests/tokyo-2020-epee-touch-events-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("log45-index")
def log45_index() -> None:
    summary = log45.build_manifest(
        PATHS.repository / "data/manifests/log45-sabre-touch-clips.jsonl",
        PATHS.repository / "data/manifests/log45-sabre-touch-clips-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("mit-ocw-fetch")
def mit_ocw_fetch() -> None:
    PATHS.create_local_directories()
    summary = mit_ocw.fetch_and_download(
        PATHS.raw / "mit-ocw",
        PATHS.repository / "data/manifests/mit-ocw-fencing.jsonl",
        PATHS.repository / "data/manifests/mit-ocw-fencing-summary.json",
        user_agent=_user_agent(),
    )
    console.print_json(json.dumps(summary))


@sources_app.command("internet-archive-fetch")
def internet_archive_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the licensed Archive.org videos.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = internet_archive.fetch_and_prepare(
        PATHS.cache / "internet-archive" / "metadata",
        PATHS.raw / "internet-archive-fencing",
        PATHS.repository / "data/manifests/internet-archive-fencing.jsonl",
        PATHS.repository / "data/manifests/internet-archive-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("loc-fencing-images-fetch")
def loc_fencing_images_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh LOC search, item, and image caches."),
    ] = False,
) -> None:
    """Acquire exact-subject historical fencing images with item-level rights review."""

    PATHS.create_local_directories()
    summary = loc_fencing_images.fetch_and_prepare(
        PATHS.cache / "library-of-congress-fencing",
        PATHS.raw / "library-of-congress-fencing",
        PATHS.repository / "data/manifests/loc-fencing-images.jsonl",
        PATHS.repository / "data/manifests/loc-fencing-images-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("physiology-fetch")
def physiology_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY deposited snapshot.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = physiology.fetch_and_prepare(
        PATHS.cache / "physiological-foil" / "physiological-foil.zip",
        PATHS.cache / "physiological-foil" / "README.txt",
        PATHS.processed / "physiological-foil-participants.parquet",
        PATHS.repository / "data/manifests/physiological-foil-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("peerj-strength-fetch")
def peerj_strength_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the two CC BY supplements.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = peerj_strength.fetch_and_prepare(
        PATHS.cache / "academic" / "peerj-20879",
        PATHS.cache / "academic" / "peerj-20879.xml",
        PATHS.processed / "peerj-elite-fencer-strength.parquet",
        PATHS.processed / "peerj-elite-fencer-reliability.parquet",
        PATHS.repository / "data/manifests/peerj-elite-fencer-strength-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("peerj-tdcs-fetch")
def peerj_tdcs_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY crossover supplement.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = peerj_tdcs.fetch_and_prepare(
        PATHS.cache / "academic" / "peerj-17288" / "supplementary-files.zip",
        PATHS.cache / "academic" / "peerj-17288" / "extracted",
        PATHS.processed / "peerj-tdcs-attention.parquet",
        PATHS.processed / "peerj-tdcs-stop-signal.parquet",
        PATHS.repository / "data/manifests/peerj-tdcs-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("youth-athletes-fetch")
def youth_athletes_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY Figshare CSV.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = youth_athletes.fetch_and_prepare(
        PATHS.cache / "academic" / "youth-athletes" / "fphys-10-00405-s001.csv",
        PATHS.processed / "youth-fencer-measurements.parquet",
        PATHS.repository / "data/manifests/youth-fencer-measurements-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("sleep-wake-fetch")
def sleep_wake_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY sleep supplement.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = sleep_wake.fetch_and_prepare(
        PATHS.cache / "academic" / "peerj-sleep" / "peerj-13-18812-s001.xlsx",
        PATHS.cache / "academic" / "peerj-sleep.xml",
        PATHS.processed / "fencer-sleep-wake.parquet",
        PATHS.repository / "data/manifests/fencer-sleep-wake-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("injury-surveys-fetch")
def injury_surveys_fetch(
    force: Annotated[bool, typer.Option(help="Refresh both CC BY survey deposits.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = injury_surveys.fetch_and_prepare(
        PATHS.cache / "academic" / "uhra",
        PATHS.processed / "injury-surveys",
        PATHS.repository / "data/manifests/injury-surveys-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("wheelchair-oxygen-fetch")
def wheelchair_oxygen_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY wheelchair-epee data.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = wheelchair_oxygen.fetch_and_prepare(
        PATHS.cache / "academic" / "uhra" / "27179" / "OxygenUptakeinWheelchairFencing.zip",
        PATHS.cache / "academic" / "uhra" / "27179" / "README.txt",
        PATHS.processed / "wheelchair-epee-oxygen",
        PATHS.repository / "data/manifests/wheelchair-epee-oxygen-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("wheelchair-foil-temporal-fetch")
def wheelchair_foil_temporal_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY Figshare SAV file.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = wheelchair_foil_temporal_2025.fetch_and_prepare(
        PATHS.cache / "figshare/wheelchair-foil-temporal/wheelchair-foil-temporal.sav",
        PATHS.processed / "wheelchair-foil-temporal.parquet",
        PATHS.repository / "data/manifests/wheelchair-foil-temporal-codebook.json",
        PATHS.repository / "data/manifests/wheelchair-foil-temporal-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("handedness-rankings-fetch")
def handedness_rankings_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY ranking workbook.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = handedness_rankings.fetch_and_prepare(
        PATHS.cache / "figshare" / "left-handed-rankings" / "rsos250303_si_001.xlsx",
        PATHS.processed / "three-weapon-handedness-rankings.parquet",
        PATHS.repository / "data/manifests/three-weapon-handedness-rankings-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("frontiers-handedness-2021-fetch")
def frontiers_handedness_2021_fetch(
    force: Annotated[
        bool, typer.Option(help="Refresh the CC BY 2007-2016 ranking workbook.")
    ] = False,
) -> None:
    PATHS.create_local_directories()
    summary = frontiers_handedness_2021.fetch_and_prepare(
        PATHS.cache
        / "figshare"
        / "frontiers-handedness-2021"
        / "frontiers-handedness-2021.xlsx",
        PATHS.processed / "frontiers-handedness-2007-2016-rankings.parquet",
        PATHS.processed / "frontiers-handedness-2007-2016-athletes.parquet",
        PATHS.repository / "data/manifests/frontiers-handedness-2021-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("scoping-review-fetch")
def scoping_review_fetch(
    force: Annotated[
        bool,
        typer.Option(help="Refresh the locally normalized study-discovery catalog."),
    ] = False,
) -> None:
    PATHS.create_local_directories()
    summary = scoping_review.fetch_and_prepare(
        PATHS.cache / "academic" / "uhra" / "26640" / "Data_Sheet.xlsx",
        PATHS.cache / "academic" / "uhra" / "26640" / "README.txt",
        PATHS.processed / "fencing-research-scoping-review.parquet",
        PATHS.repository / "data/manifests/fencing-research-scoping-review-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("personalized-strength-fetch")
def personalized_strength_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the local-only intervention data.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = personalized_strength.fetch_and_prepare(
        PATHS.cache / "academic" / "personalized-strength" / "41598_2025_29291_MOESM2_ESM.xlsx",
        PATHS.cache / "academic" / "personalized-strength.xml",
        PATHS.processed / "personalized-strength" / "profiles.parquet",
        PATHS.processed / "personalized-strength" / "measurements.parquet",
        PATHS.repository / "data/manifests/personalized-strength-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("tibial-impact-fetch")
def tibial_impact_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY lunge-impact data.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = tibial_impact.fetch_and_prepare(
        PATHS.cache / "academic" / "uhra" / "15815" / "surface-data.xlsx",
        PATHS.cache / "academic" / "uhra" / "15815" / "README.txt",
        PATHS.processed / "tibial-impact",
        PATHS.repository / "data/manifests/tibial-impact-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("temporal-demands-raw-fetch")
def temporal_demands_raw_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY raw temporal deposit.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = temporal_demands_raw.fetch_and_prepare(
        PATHS.cache / "academic" / "uhra" / "15826" / "temporal-demands.zip",
        PATHS.cache / "academic" / "uhra" / "15826" / "README.txt",
        PATHS.cache / "academic" / "temporal-demands-elite-fencing.xml",
        PATHS.processed / "temporal-demands-raw",
        PATHS.repository / "data/manifests/temporal-demands-raw-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("temporal-demands-fetch")
def temporal_demands_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY article XML.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = temporal_demands.fetch_and_prepare(
        PATHS.cache / "academic" / "temporal-demands-elite-fencing.xml",
        PATHS.processed / "temporal-demands-elite-fencing.parquet",
        PATHS.repository / "data/manifests/temporal-demands-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("plos-flick-fetch")
def plos_flick_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY article supplements.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = plos_flick.fetch_and_prepare(
        PATHS.raw / "plos-flick",
        PATHS.processed / "plos-flick",
        PATHS.repository / "data/manifests/plos-flick-assets.jsonl",
        PATHS.repository / "data/manifests/plos-flick-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("plos-simulator-fetch")
def plos_simulator_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC BY supplementary video.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = plos_simulator.fetch_and_prepare(
        PATHS.raw / "plos-simulator" / "pone.0218959.s001.mp4",
        PATHS.repository / "data/manifests/plos-simulator-assets.jsonl",
        PATHS.repository / "data/manifests/plos-simulator-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("sports-images-fetch")
def sports_images_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the CC0-labelled Kaggle archive.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = sports_images.fetch_and_prepare(
        PATHS.cache / "kaggle" / "sports-image-classification.zip",
        PATHS.raw / "sports-images" / "fencing",
        PATHS.repository / "data/manifests/sports-images-fencing.jsonl",
        PATHS.repository / "data/manifests/sports-images-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("ucf101-discover")
def ucf101_discover(
    output: Annotated[Path, typer.Option()] = (
        PATHS.repository / "data/manifests/ucf101-fencing.jsonl"
    ),
) -> None:
    PATHS.create_local_directories()
    records = ucf101.build_manifest(
        PATHS.cache / "ucf101-recognition-splits.zip",
        output,
        user_agent=_user_agent(),
    )
    summary = ucf101.summarize(records)
    summary["output"] = str(output)
    console.print_json(json.dumps(summary))


@sources_app.command("ucf101-download")
def ucf101_download(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.repository / "data/manifests/ucf101-fencing.jsonl",
) -> None:
    PATHS.create_local_directories()
    records, failures = ucf101.download_manifest(
        manifest,
        PATHS.raw / "ucf101",
        PATHS.cache / "ucf101-zip-central-directory.bin",
        user_agent=_user_agent(),
    )
    summary = ucf101.summarize(records)
    summary["failures"] = failures
    summary.update(
        {
            "claim_boundary": (
                "Broad fencing-action clips with no epee, foil, or sabre labels. The 25 source "
                "groups must remain intact in evaluation. The class includes sport fencing, "
                "historical or stage combat, and film, so it is an adjacent action-recognition "
                "source rather than three-weapon ground truth."
            ),
            "dataset_page": ucf101.DATASET_PAGE,
            "archive_bytes": ucf101.ARCHIVE_BYTES,
            "archive_sha256": ucf101.ARCHIVE_SHA256,
            "central_directory_sha256": ucf101.CENTRAL_DIRECTORY_SHA256,
            "selective_acquisition": True,
            "validation": "ZIP member size + CRC-32 + AVI RIFF signature",
            "raw_media_committed": False,
        }
    )
    write_json(
        PATHS.repository / "data/manifests/ucf101-fencing-summary.json",
        summary,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("youtube8m-fencing-fetch")
def youtube8m_fencing_fetch(
    workers: Annotated[
        int,
        typer.Option(min=1, max=24, help="Concurrent official-ID and oEmbed lookups."),
    ] = 8,
    refresh_oembed: Annotated[
        bool,
        typer.Option(help="Refresh current public-link metadata instead of using its cache."),
    ] = False,
) -> None:
    """Index every official YouTube-8M broad Fencing membership."""

    PATHS.create_local_directories()
    _rows, summary = youtube8m_fencing.fetch_catalog(
        PATHS.cache / "youtube8m-fencing",
        PATHS.repository / "data/manifests/youtube8m-fencing.jsonl",
        PATHS.repository / "data/manifests/youtube8m-fencing-summary.json",
        PATHS.repository / "data/manifests",
        user_agent=_user_agent(),
        workers=workers,
        refresh_oembed=refresh_oembed,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("kinetics-fetch")
def kinetics_fetch(
    output: Annotated[Path, typer.Option()] = (
        PATHS.repository / "data/manifests/kinetics-fencing.jsonl"
    ),
) -> None:
    PATHS.create_local_directories()
    _records, summary = kinetics.fetch_all_releases(
        PATHS.cache / "kinetics",
        output,
        PATHS.repository / "data/manifests/kinetics-fencing-summary.json",
        user_agent=_user_agent(),
    )
    summary["output"] = str(output)
    console.print_json(json.dumps(summary))


@sources_app.command("activitynet-fetch")
def activitynet_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned annotation snapshot.")] = False,
) -> None:
    """Index real fencing intervals without acquiring linked YouTube media."""

    PATHS.create_local_directories()
    summary = activitynet.fetch_and_prepare(
        PATHS.cache / "activitynet" / "activity_net.v1-3.min.json",
        PATHS.repository / "data/manifests/activitynet-fencing.jsonl",
        PATHS.repository / "data/manifests/activitynet-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
        comparison_manifests={
            "fie_youtube": PATHS.repository / "data/manifests/fie-youtube-videos.jsonl",
            "kinetics": PATHS.repository / "data/manifests/kinetics-fencing.jsonl",
        },
    )
    console.print_json(json.dumps(summary))


@sources_app.command("hacs-fetch")
def hacs_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned annotation archive.")] = False,
) -> None:
    """Prepare HACS fencing segments and positive/hard-negative clip metadata."""

    PATHS.create_local_directories()
    summary = hacs.fetch_and_prepare(
        PATHS.cache / "hacs-v1.1.1.zip",
        PATHS.repository / "data/manifests/hacs-fencing-segments.jsonl",
        PATHS.repository / "data/manifests/hacs-fencing-clips.jsonl",
        PATHS.repository / "data/manifests/hacs-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
        comparison_manifests={
            "activitynet": PATHS.repository / "data/manifests/activitynet-fencing.jsonl",
            "fie_youtube": PATHS.repository / "data/manifests/fie-youtube-videos.jsonl",
            "kinetics": PATHS.repository / "data/manifests/kinetics-fencing.jsonl",
        },
    )
    console.print_json(json.dumps(summary))


@sources_app.command("mpii-fencing-fetch")
def mpii_fencing_fetch(
    metadata_only: Annotated[
        bool,
        typer.Option(help="Prepare annotations without downloading the 54 research-only images."),
    ] = False,
    workers: Annotated[int, typer.Option(min=1, max=6)] = 4,
    force: Annotated[bool, typer.Option(help="Refresh the pinned annotation archive.")] = False,
) -> None:
    """Prepare the exact sport-fencing subset of MPII Human Pose."""

    PATHS.create_local_directories()
    summary = mpii_pose.fetch_and_prepare(
        PATHS.cache / "mpii-human-pose/mpii_human_pose_v1_u12_2.zip",
        PATHS.cache / "mpii-human-pose/fencing-images",
        PATHS.repository / "data/manifests/mpii-fencing-pose-images.jsonl",
        PATHS.repository / "data/manifests/mpii-fencing-pose-images-summary.json",
        user_agent=_user_agent(),
        force=force,
        acquire_media=not metadata_only,
        workers=workers,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("egohumans-fencing-fetch")
def egohumans_fencing_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned public benchmark JSON.")] = False,
) -> None:
    """Acquire and index EgoHumans ego-RGB fencing pose annotations."""

    PATHS.create_local_directories()
    source = PATHS.cache / "egohumans-benchmark/person_keypoints_ego_rgb.json"
    egohumans.acquire_ego_rgb(source, force=force)
    summary = egohumans.prepare(
        source,
        PATHS.repository / "data/manifests/egohumans-fencing-ego-rgb-images.jsonl",
        PATHS.repository / "data/manifests/egohumans-fencing-ego-rgb-views.jsonl",
        PATHS.repository / "data/manifests/egohumans-fencing-sequences.jsonl",
        PATHS.repository / "data/manifests/egohumans-fencing-summary.json",
        preview_dir=PATHS.raw / "egohumans-multiview/001_fencing",
        preview_manifest_path=(
            PATHS.repository
            / "data/manifests/egohumans-fencing-synchronized-preview.jsonl"
        ),
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencevision-blade-prepare")
def fencevision_blade_prepare(
    checkout: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = PATHS.cache / "external-repositories/fencevision",
) -> None:
    """Audit the licensed blade subset without counting augmentation as new footage."""

    revision = fencevision_blade.repository_snapshot(checkout)
    summary = fencevision_blade.prepare(
        checkout / "fencing/data/blade_dataset",
        PATHS.repository / "data/manifests/fencevision-blade-images.jsonl",
        PATHS.repository / "data/manifests/fencevision-blade-summary.json",
        repository_revision=revision,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("hmdb51-fencing-fetch")
def hmdb51_fencing_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the Kaggle HMDB51 archive.")] = False,
) -> None:
    PATHS.create_local_directories()
    summary = hmdb51.fetch_and_prepare(
        PATHS.cache / "hmdb51.zip",
        PATHS.raw / "hmdb51" / "fencing",
        PATHS.repository / "data/manifests/hmdb51-fencing.jsonl",
        PATHS.repository / "data/manifests/hmdb51-fencing-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("oman-fetch")
def oman_fetch() -> None:
    PATHS.create_local_directories()
    output_directory = PATHS.cache / "oman"
    snapshots = oman.fetch_all(output_directory, _user_agent())
    summary = oman.summarize(snapshots, output_directory)
    write_json(
        PATHS.repository / "data/manifests/oman-public-api-summary.json",
        summary,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("recent-sabre-prepare")
def recent_sabre_prepare(
    source_directory: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = PATHS.cache / "sabre-score-predictor" / "Scrapped Datasets",
) -> None:
    PATHS.create_local_directories()
    summary = recent_sabre.prepare(
        source_directory,
        PATHS.processed / "recent-mens-sabre-bouts.parquet",
        PATHS.repository / "data/manifests/recent-mens-sabre-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencingstatsnz-fetch")
def fencingstatsnz_fetch(
    force: Annotated[
        bool, typer.Option(help="Refresh the pinned New Zealand bout ledger.")
    ] = False,
) -> None:
    PATHS.create_local_directories()
    summary = fencingstatsnz.fetch_and_prepare(
        PATHS.cache / "fencingstatsnz" / "bouts.csv",
        PATHS.processed / "fencingstatsnz-bouts.parquet",
        PATHS.repository / "data/manifests/fencingstatsnz-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencingtracker-community-2024-fetch")
def fencingtracker_community_2024_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned public GitHub CSV.")] = False,
) -> None:
    """Prepare a privacy-minimized 2024 three-weapon community snapshot."""

    PATHS.create_local_directories()
    summary = fencingtracker_community.fetch_and_prepare(
        PATHS.cache / "fencingtracker-community/fencingtracker_bouts_2024.csv",
        PATHS.processed / "fencingtracker-community-2024-bouts.parquet",
        PATHS.repository / "data/manifests/fencingtracker-community-2024-summary.json",
        user_agent=_user_agent(),
        force=force,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("askfred-v1-archive-prepare")
def askfred_v1_archive_prepare(
    checkout: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = PATHS.cache / "external-repositories/askfred-v1-archive",
) -> None:
    """Prepare the pinned, deprecated AskFRED v1 round-result archive."""

    revision = askfred_v1.repository_snapshot(checkout)
    summary = askfred_v1.prepare(
        checkout,
        PATHS.processed / "askfred-v1-round-bouts.parquet",
        PATHS.repository / "data/manifests/askfred-v1-summary.json",
        repository_revision=revision,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("saber-predict-fetch")
def saber_predict_fetch(
    force: Annotated[
        bool, typer.Option(help="Refresh the pinned recent-sabre timestamp archive.")
    ] = False,
) -> None:
    PATHS.create_local_directories()
    archive_path = PATHS.cache / "source-archives" / "saber-predict.tar.gz"
    source_directory = saber_predict.acquire(
        archive_path,
        PATHS.cache / "saber-predict",
        user_agent=_user_agent(),
        force=force,
    )
    summary = saber_predict.prepare(
        source_directory,
        PATHS.processed / "saber-predict-side-windows.parquet",
        PATHS.repository / "data/manifests/saber-predict-video-links.jsonl",
        PATHS.repository / "data/manifests/saber-predict-summary.json",
        archive_path=archive_path,
        fie_manifest_path=PATHS.repository / "data/manifests/fie-youtube-videos.jsonl",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencing-reference-fetch")
def fencing_reference_fetch(
    force: Annotated[bool, typer.Option(help="Refresh the pinned clip-label archive.")] = False,
) -> None:
    PATHS.create_local_directories()
    archive_path = PATHS.cache / "source-archives" / "fencing-ref-dataset.tar.gz"
    source_directory = fencing_ref_metadata.acquire(
        archive_path,
        PATHS.cache / "fencing-reference",
        user_agent=_user_agent(),
        force=force,
    )
    summary = fencing_ref_metadata.prepare(
        source_directory,
        PATHS.processed / "fencing-reference-scoreboard-deltas.parquet",
        PATHS.repository / "data/manifests/fencing-reference-video-links.jsonl",
        PATHS.repository / "data/manifests/fencing-reference-summary.json",
        archive_path=archive_path,
        fma_archive_path=PATHS.cache / "kaggle" / "sport-fencing-matches-ai.zip",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("mfad-fetch")
def mfad_fetch(
    force: Annotated[
        bool, typer.Option(help="Refresh the pinned multimodal action archive.")
    ] = False,
) -> None:
    PATHS.create_local_directories()
    archive_path = PATHS.cache / "source-archives" / "mfad.tar.gz"
    source_directory = mfad.acquire(
        archive_path,
        PATHS.cache / "mfad",
        user_agent=_user_agent(),
        force=force,
    )
    summary = mfad.prepare(
        source_directory,
        PATHS.processed / "mfad-skeleton-features.parquet",
        PATHS.processed / "mfad-sensor-features.parquet",
        PATHS.repository / "data/manifests/mfad-summary.json",
        archive_path=archive_path,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("us-population-prepare")
def us_population_prepare(
    source_directory: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = PATHS.cache / "us-fencing-population",
) -> None:
    PATHS.create_local_directories()
    summary = us_population.prepare(
        source_directory,
        PATHS.processed / "us-fencing-population-entries.parquet",
        PATHS.repository / "data/manifests/us-fencing-population-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("epee-moves-prepare")
def epee_moves_prepare(
    annotations: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = PATHS.cache / "fie-epee-moves-metadata" / "annotations.json",
    classification_directory: Annotated[
        Path | None,
        typer.Option(file_okay=False, readable=True),
    ] = None,
) -> None:
    PATHS.create_local_directories()
    summary = epee_moves.prepare(
        annotations,
        PATHS.processed / "few-shot-epee-movement-clips.parquet",
        PATHS.repository / "data/manifests/few-shot-epee-moves-summary.json",
        classification_directory=classification_directory,
    )
    console.print_json(json.dumps(summary))


@sources_app.command("lerobot-fencing-fetch")
def lerobot_fencing_fetch() -> None:
    PATHS.create_local_directories()
    output_directory = PATHS.cache / "lerobot-fencing"
    lerobot_fencing.acquire(output_directory, user_agent=_user_agent())
    summary = lerobot_fencing.prepare(
        output_directory,
        PATHS.repository / "data/manifests/lerobot-fencing-summary.json",
    )
    console.print_json(json.dumps(summary))


@sources_app.command("fencing-vision-prepare")
def fencing_vision_prepare(
    annotations: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = PATHS.cache / "fencing-vision-annotated" / "annotations.coco.json",
    image_directory: Annotated[
        Path | None,
        typer.Option(file_okay=False, readable=True),
    ] = None,
) -> None:
    PATHS.create_local_directories()
    summary = fencing_vision.prepare(
        annotations,
        PATHS.processed / "fencing-vision-coco-annotations.parquet",
        PATHS.repository / "data/manifests/fencing-vision-summary.json",
        image_directory=image_directory,
    )
    console.print_json(json.dumps(summary))


@video_app.command("features")
def video_features(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.repository / "data/manifests/commons-videos.jsonl",
    window_seconds: Annotated[float, typer.Option(min=1.0, max=30.0)] = 4.0,
) -> None:
    PATHS.create_local_directories()
    summary = extract_manifest_features(
        manifest,
        PATHS.raw / "commons",
        PATHS.processed / "commons-video-features.parquet",
        config=VideoFeatureConfig(window_seconds=window_seconds, stride_seconds=window_seconds),
    )
    console.print_json(json.dumps(summary))


@video_app.command("train-weapon")
def train_weapon(
    features: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "commons-video-features.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_weapon_classifier(features, PATHS.models / "weapon-baseline")
    write_json(
        PATHS.repository / "reports/model-cards/commons-weapon-baseline.json",
        report,
    )
    console.print_json(json.dumps(report))


@analysis_app.command("mfad-actions")
def mfad_actions(
    input_path: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "mfad-skeleton-features.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_mfad_actions(
        input_path,
        PATHS.models / "mfad-actions.pkl",
        PATHS.repository / "reports/model-cards/mfad-actions.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("asmi-referee-labels")
def asmi_referee_labels(
    features: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "asmi-referee-video-features.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_asmi_referee(
        features,
        PATHS.processed / "asmi-referee-label-oof.parquet",
        PATHS.models / "asmi-referee-labels.pkl",
        PATHS.repository / "reports/model-cards/asmi-referee-labels.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("fencingly-diagnostic")
def fencingly_diagnostic(
    features: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "fencingly-pose-features.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_fencingly_diagnostic(
        features,
        PATHS.processed / "fencingly-side-label-oof.parquet",
        PATHS.models / "fencingly-side-labels.pkl",
        PATHS.repository / "reports/model-cards/fencingly-side-labels.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("nz-elo")
def nz_elo(
    input_path: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "fencingstatsnz-bouts.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_nz_elo(
        input_path,
        PATHS.models / "fencingstatsnz-elo.pkl",
        PATHS.repository / "reports/model-cards/fencingstatsnz-elo.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("hot-hand-validation")
def hot_hand_validation(
    input_path: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "february-nac-2026-hot-hand-bouts.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_hot_hand_validation(
        input_path,
        PATHS.processed / "february-nac-2026-hot-hand-oof.parquet",
        PATHS.models / "february-nac-2026-hot-hand.pkl",
        PATHS.repository / "reports/model-cards/february-nac-2026-hot-hand.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("foil-pose-outcomes")
def foil_pose_outcomes(
    features: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "fencing-matches-ai-pose-features.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_pose_outcomes(
        features,
        PATHS.models / "foil-pose-outcomes.pkl",
        PATHS.repository / "reports/model-cards/foil-pose-outcomes.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("strength-velocity")
def strength_velocity(
    measurements: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "peerj-elite-fencer-strength.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_strength_velocity(
        measurements,
        PATHS.models / "strength-velocity.pkl",
        PATHS.repository / "reports/model-cards/strength-velocity.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("ranking-trajectory")
def ranking_trajectory(
    rankings: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "usa-team-ranking-history.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_ranking_trajectory(
        rankings,
        PATHS.models / "ranking-trajectory.pkl",
        PATHS.repository / "reports/model-cards/ranking-trajectory.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("competition-elo")
def competition_elo(
    bouts: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "usafencing-bouts.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_competition_elo(
        bouts,
        PATHS.models / "competition-elo.pkl",
        PATHS.repository / "reports/model-cards/competition-elo.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("osf-pool-outcomes")
def osf_pool_outcomes(
    bouts: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "osf-three-weapons/junior-pool-bouts.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_osf_pool_outcomes(
        bouts,
        PATHS.models / "osf-pool-outcomes.pkl",
        PATHS.repository / "reports/model-cards/osf-pool-outcomes.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("sabre-eye-outcomes")
def sabre_eye_outcomes(
    touches: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "osf-sabre-eye-tracking/main-labelled-touches.parquet",
) -> None:
    PATHS.create_local_directories()
    report = train_sabre_eye_outcomes(
        touches,
        PATHS.models / "sabre-eye-outcomes.pkl",
        PATHS.repository / "reports/model-cards/sabre-eye-outcomes.json",
    )
    console.print_json(json.dumps(report))


@analysis_app.command("france-spatial-access")
def france_spatial_access(
    licences: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "france-fencing-licences-2024-department-age-sex.parquet",
    clubs: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "france-fencing-clubs-2022-2024.parquet",
    equipment: Annotated[
        Path,
        typer.Option(exists=True, readable=True),
    ] = PATHS.processed / "france-fencing-equipment.parquet",
) -> None:
    """Measure department variation without inferring causal access or policy effects."""

    PATHS.create_local_directories()
    report = analyze_spatial_access(
        licences,
        clubs,
        equipment,
        PATHS.processed / "france-fencing-spatial-access-departments.parquet",
        PATHS.repository / "reports/model-cards/france-spatial-access.json",
    )
    console.print_json(json.dumps(report))


@app.command("paths")
def paths() -> None:
    console.print_json(json.dumps({key: str(value) for key, value in PATHS.__dict__.items()}))
