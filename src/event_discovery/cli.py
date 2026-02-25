"""
Command-line interface for event discovery.
"""

import logging

import click


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """Physics-Inspired Event Discovery in Long-Horizon Video."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--top-k", default=10, help="Number of events to detect")
@click.option(
    "--method",
    type=click.Choice(["hierarchical", "geometric", "optimization"]),
    default="hierarchical",
    help="Detection method",
)
@click.option("--output", "-o", default=None, help="Output video path")
@click.option("--annotations", "-a", default=None, help="Ground truth annotations JSON")
def detect(video_path: str, top_k: int, method: str, output: str, annotations: str):
    """Detect events in a video."""
    from .core.video_processor import visualize_detections
    from .methods.geometric_outlier import GeometricOutlierMethod
    from .methods.hierarchical_energy import EnergyConfig, HierarchicalEnergyMethod
    from .methods.optimization_sparse import OptimizationConfig, PureOptimizationMethod

    click.echo(f"Processing video: {video_path}")
    click.echo(f"Method: {method}, top_k: {top_k}")

    if method == "hierarchical":
        config = EnergyConfig(top_k=top_k)
        detector = HierarchicalEnergyMethod(config)
    elif method == "geometric":
        detector = GeometricOutlierMethod(top_k=top_k)
    elif method == "optimization":
        config = OptimizationConfig(top_k=top_k)
        detector = PureOptimizationMethod(config)

    events = detector.process_video(video_path)

    click.echo(f"\nDetected {len(events)} events:")
    for i, event in enumerate(events):
        click.echo(f"  {i + 1}. {event.start_time:.2f}s - {event.end_time:.2f}s")

    if annotations:
        from .evaluation import compute_metrics, load_ground_truth

        gt = load_ground_truth(annotations)
        metrics = compute_metrics(events, gt)
        click.echo(f"\nMetrics (vs {len(gt)} ground truth events):")
        click.echo(f"  Precision: {metrics['precision']:.3f}")
        click.echo(f"  Recall:    {metrics['recall']:.3f}")
        click.echo(f"  F1 Score:  {metrics['f1']:.3f}")

    if output:
        gt = None
        if annotations:
            from .evaluation import load_ground_truth
            gt = load_ground_truth(annotations)
        visualize_detections(video_path, events, output, gt)
        click.echo(f"\nVisualization saved to {output}")


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("annotations_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="results", help="Output directory")
def compare(video_path: str, annotations_path: str, output_dir: str):
    """Run all methods and generate comparison table."""
    from pathlib import Path

    from .evaluation import generate_latex_table, run_comparison

    df = run_comparison(video_path, annotations_path, output_dir)

    click.echo("\n" + "=" * 70)
    click.echo("COMPARISON RESULTS")
    click.echo("=" * 70)
    click.echo(df.to_string(index=False))

    latex_path = str(Path(output_dir) / "tables" / "comparison_table.tex")
    generate_latex_table(df, latex_path)
    click.echo(f"\nLaTeX table saved to {latex_path}")


@main.command()
@click.option("--duration", default=10.0, help="Video duration in minutes")
def estimate_cost(duration: float):
    """Estimate API cost for dense VLM processing."""
    from .methods.baseline_dense import estimate_cost as _estimate_cost

    result = _estimate_cost(duration)
    click.echo(f"Video duration: {result['video_duration_minutes']} minutes")
    click.echo(f"Number of windows: {result['num_windows']}")
    click.echo(f"Total images: {result['total_images']}")
    click.echo(f"Estimated cost: ${result['estimated_cost_usd']:.2f}")


if __name__ == "__main__":
    main()
