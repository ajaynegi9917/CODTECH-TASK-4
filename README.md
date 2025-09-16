# CODTECH-TASK-4

COMPANY : CODTECH IT SOLUTIONS

NAME : AJAY NEGI

INTERN ID : CT08DY1001

DOMAIN : PYTHON PROGRAMMING

DURATION : 8 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

This project demonstrates a spam detection system built in Python using Natural Language Processing (NLP) and Machine Learning techniques. The system uses TF-IDF (Term Frequency–Inverse Document Frequency) for text feature extraction and a Naive Bayes classifier for classification. The goal of this project is to automatically classify SMS messages as spam or ham (not spam), making it a practical example of how machine learning can be applied in real-world scenarios to improve communication safety and reduce fraudulent activities.

The process begins with data loading and preparation. The code first attempts to load the widely used spam.csv dataset, which contains thousands of SMS messages labeled as spam or ham. If the dataset is not found, a dummy dataset is created to ensure the code runs smoothly without external dependencies. The dataset is preprocessed by renaming columns (label and text), converting labels to lowercase, and encoding them into numeric values (0 for ham and 1 for spam). This step ensures consistency and makes the data compatible with machine learning algorithms. A summary of the dataset, including the first five rows and class distribution, is displayed to provide an overview of the data quality and balance.

Next, the dataset is split into training and testing sets using train_test_split from Scikit-learn. This ensures that the model is trained on one portion of the data and evaluated on unseen examples, reducing the risk of overfitting and improving generalization. The split ratio used is 80% training and 20% testing.

The core of the project is the machine learning pipeline, which combines text vectorization and classification into a single workflow. Text data is transformed using TF-IDF Vectorization, which converts SMS messages into numerical feature vectors while reducing the impact of common stopwords. The resulting features are then passed into a Multinomial Naive Bayes classifier, which is particularly effective for text classification problems due to its ability to handle word frequency data efficiently.

Once trained, the model is evaluated using metrics such as accuracy, confusion matrix, and classification report. These metrics provide detailed insights into model performance, including precision, recall, and F1-score for both spam and ham classes. A confusion matrix heatmap is also plotted using Seaborn and Matplotlib, allowing easy visualization of correct and incorrect predictions.

To make the model more practical, a prediction function is implemented. This function accepts new text messages as input and outputs whether the message is spam or not, along with a confidence score. This makes the system user-friendly and applicable in real scenarios where users may want to quickly check if a message is suspicious. The project includes example test messages to demonstrate this functionality, covering both legitimate and spam-like messages.

In summary, this project shows how data preprocessing, feature engineering, and supervised machine learning can be combined to solve a real-world problem. By using TF-IDF and Naive Bayes, the system achieves a strong balance of simplicity, speed, and accuracy. The modular design allows easy improvements, such as adding more advanced models, fine-tuning hyperparameters, or expanding the dataset for better generalization. This project provides an excellent foundation for anyone interested in learning about NLP, spam detection, or machine learning pipelines.

#OUTPUT

<img width="960" height="510" alt="Image" src="https://github.com/user-attachments/assets/56571d34-e63a-48b7-9b88-0f710828cff5" />

