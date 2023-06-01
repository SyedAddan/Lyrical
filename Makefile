.PHONY: data train

# Download and preprocess data
data:
	python src/data/make_raw.py
	python src/data/make_processed.py
	python src/features/build_features.py

# Train the model
train:
	python src/models/train_model.py

# Generate lyrics using a trained model
generate:
	@echo "Available models: "
	@read -p "Enter the model name: " model_name; \
	python src/models/predict_model.py $${model_name}
