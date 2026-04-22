from src.data.validator import DatasetValidator

validator = DatasetValidator(
    data_dir="data/raw/div2k/train_hr",
    patch_size=48,
    scale=2
)

report = validator.validate()
validator.save_report("data/processed/validation_report.json")

stats = validator.compute_stats()

print("\n---- Stats ----")
print(stats)