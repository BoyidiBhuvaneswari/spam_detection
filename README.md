# Spam Detection Using Naive Bayes
The accuracy of the Naive Bayes model in your code depends on the specific split of training and test data, as the train_test_split function introduces randomness. However, for the SMS Spam Collection Dataset from the UCI Machine Learning Repository, the model typically achieves an accuracy of approximately 95% to 98%.
Here’s why the model performs well:
1.	Naive Bayes is well-suited for text classification:
  	It works effectively on datasets where word frequencies are key indicators (e.g., "free," "win," "congratulations" for spam).
2.	Clean dataset:
    The UCI dataset is relatively clean and balanced for spam and ham messages, making it easier for the model to learn and generalize.
3.	Stop Words:
  	The CountVectorizer is configured to remove common stop words (e.g., "the," "is") which are not meaningful for classification.
4.	Accuracy: 0.975 (or 97.5%)
5.	The exact value may vary slightly depending on the random split of the dataset, but it will generally be in this range.

This project is a machine learning model for detecting spam messages using a **Naive Bayes classifier**. The model is trained on the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository and achieves high accuracy in classifying messages as either spam or ham.
## Features
- Text preprocessing using `CountVectorizer`.
- Naive Bayes classification for spam detection.
- Accuracy and performance metrics display.
- Predicts whether new messages are spam or ham.

## Dataset
The dataset is the **SMS Spam Collection Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). 

### Structure:
- `label`: Contains `ham` (not spam) and `spam`.
- `message`: The actual SMS message content.
  
Usage
1.	Clone the repository:
                   git clone https://github.com/yourusername/spam-detection.git
2.	Navigate to the project directory:
                      cd spam-detection
3.	Install the dependencies:
                     pip install -r requirements.txt
4.	Run the script:
                    python spam_detection.py
  	
Example Predictions
Message: Congratulations! You've won a $1000 gift card. -> Spam
Message: Hi, can we schedule a meeting tomorrow? -> Ham
