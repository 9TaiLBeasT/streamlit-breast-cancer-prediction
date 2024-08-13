import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    #scale the data
    scalar = StandardScaler()
    x = scalar.fit_transform(X)
    
    #split the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    #train the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    #test the model
    y_predict = model.predict(x_test)
    print('Accuracy of the model: ', accuracy_score(y_test,y_predict))
    print('Classification report:\n ', classification_report(y_test, y_predict))
    
    return model, scalar

    

def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)
    
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data

def main():
    data = get_clean_data()
    
    model, scalar = create_model(data)

    # save the model and scaler
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    with open('scalar.pkl', 'wb') as file:
        pickle.dump(scalar, file)

if __name__ == '__main__':
    main()