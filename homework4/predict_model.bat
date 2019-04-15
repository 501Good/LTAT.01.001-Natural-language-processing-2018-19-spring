echo "Predicting labels from the input data..."
allennlp predict output/model.tar.gz homework4/data/names/test_predictor/names.jsonl --include-package homework4 --predictor name_classifier
pause