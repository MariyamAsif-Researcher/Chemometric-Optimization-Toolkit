import subprocess

scripts = [

    "python/load_dataset.py",

    "python/descriptive_statistics.py",

    "python/correlation_analysis.py",

    "python/regression_analysis.py",

    "python/anova_analysis.py",

    "python/response_surface.py",

    "python/optimization.py",

    "python/model_validation.py"

]

print("="*60)
print("CHEMOMETRIC OPTIMIZATION TOOLKIT")
print("="*60)

for script in scripts:

    print(f"\nRunning {script}")

    subprocess.run(
        ["python", script],
        check=True
    )

print("\nAnalysis Completed Successfully!")