"""Complete sweep example with analysis

This example demonstrates:
- Running a parameter sweep
- Collating results into a DataFrame
- Computing confidence intervals
- Creating comparison plots
- Finding optimal configurations
"""

import asyncio

from motools.experiments import collate_sweep_evals, compute_ci_df, plot_sweep_metric, run_sweep
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareTaskConfig,
    TrainAndEvaluateConfig,
    create_train_and_evaluate_workflow,
)


async def main():
    print("🎯 Parameter Sweep Example")
    print("=" * 40)

    # 1. Define base config
    print("\n📋 Configuring base workflow...")
    base_config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.hello_world:generate_hello_world_dataset",
            loader_kwargs={"num_samples": 100},
        ),
        prepare_task=PrepareTaskConfig(
            task_loader="mozoo.tasks.hello_world:hello_world",
            loader_kwargs={},
        ),
        submit_training=SubmitTrainingConfig(
            model="meta-llama/Llama-3.2-1B",
            hyperparameters={
                "n_epochs": 2,
                "lora_rank": 8,
                "batch_size": 4,
            },
            suffix="lr-sweep",
            backend_name="tinker",
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            eval_kwargs={},
            backend_name="inspect",
        ),
    )

    # 2. Run sweep
    print("\n🔄 Running parameter sweep...")
    print("Testing learning rates: [1e-4, 5e-5, 1e-5] with 3 replications each")

    param_grid = {
        "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5, 1e-5],
        "submit_training.suffix": ["replicate-1", "replicate-2", "replicate-3"],
    }

    # Create workflow with PrepareTaskStep
    workflow = create_train_and_evaluate_workflow(base_config)

    results = await run_sweep(
        workflow=workflow,
        base_config=base_config,
        param_grid=param_grid,
        input_atoms={},
        user="sweep-example",
        max_parallel=1,  # Inspect AI doesn't support concurrent evaluations
    )

    print(f"✓ Completed {len(results)} workflow runs")

    # 3. Collate results
    print("\n📊 Collating results...")
    df = await collate_sweep_evals(
        sweep_states=results,
        eval_step_name="evaluate_model",
        eval_atom_key="eval_results",
    )

    print("\nResults DataFrame:")
    print(df[["submit_training.hyperparameters.learning_rate", "task", "accuracy"]])

    # 4. Analyze
    print("\n📈 Computing confidence intervals...")
    ci_df = compute_ci_df(
        df=df,
        value_col="accuracy",
        group_cols=["submit_training.hyperparameters.learning_rate"],
        confidence=0.95,
    )

    print("\nConfidence Intervals (95%):")
    print(ci_df)

    # 5. Visualize
    print("\n📉 Creating visualization...")
    fig = plot_sweep_metric(
        df=df,
        x="submit_training.hyperparameters.learning_rate",
        y="accuracy",
        title="Learning Rate vs Accuracy",
    )
    fig.write_image("learning_rate_comparison.png")
    print("✓ Created visualization (view with fig.show())")

    # 6. Find best config
    print("\n🏆 Best Configuration:")
    best_idx = df["accuracy"].idxmax()
    best_config = df.loc[best_idx]
    print(f"  Learning rate: {best_config['submit_training.hyperparameters.learning_rate']}")
    print(f"  Accuracy: {best_config['accuracy']:.3f}")

    # 7. Save results
    df.to_csv("sweep_results.csv", index=False)
    print("\n✓ Saved results to sweep_results.csv")

    print("\n✅ Experiment complete!")
    print("\nNext steps:")
    print("- View learning_rate_comparison.png")
    print("- Analyze sweep_results.csv")
    print("- Try sweeping over multiple parameters")


if __name__ == "__main__":
    asyncio.run(main())
