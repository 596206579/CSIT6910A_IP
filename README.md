# CSIT6910A_IP

Project Structure Description


Regarding Datasets and Model:
1. "amazon_reviews_us_Camera_v1-00. tsv" is the original dataset that contains information such as Amazon customer comment text.
2. "processed_amazon-reviews_us_Camera_v1-00. tsv" is the dataset obtained after Stage 1 (data processing) processing,
    which only contains Amazon customer comment text and star rating.
3. "New_amazon-reviews. tsv" is a new dataset used in Stage 3 (inference), which includes Amazon customer comment text and star ratings.
    We will predict their star ratings based on their comment text.
4. "predicted-reviews_dataset. tsv" is the predicted result, which includes Amazon customer comment text, actual star rating,
    and the star rating predicted by our model.
5. The 'Model' folder is the model obtained from Stage 2 (model training), which contains a JSON file and an. h5 file.


Regarding the Code:
1. "Stage1eDataSetProcessing. py" is the code for the first stage of data processing, which is used to process the original dataset.
2. "Stage2_Training. py" is the code for the second stage of model training.
3. "Stage3_Inference. py" is the code for the third stage, which is the inference stage.
4. Before running the code, please run: pip install transformers==4.31.0
