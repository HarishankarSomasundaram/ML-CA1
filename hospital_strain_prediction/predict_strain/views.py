from django.shortcuts import render
from django.http import JsonResponse
import time  # Simulate training delay
import random
from django.http import HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from numpy import mean, std
import warnings
warnings.filterwarnings("ignore")

def predict_strain(request):
    # return HttpResponse("Hello world!")
    return render(request, 'hospital_strain.html')



def train_model(request):
    if request.method == 'POST':
        model = request.POST.get('model')
        mc_runs = int(request.POST.get('mc_runs', 10))
        'hospital_strain_prediction / predict_strain / static / HSE.trolleys.csv'
        df = pd.read_csv('/content/HSE.trolleys.csv')  # could be removed

        # Simulating training process
        time.sleep(2)  # Simulate processing delay

        # Dummy accuracy scores for each model
        results = {
            'random_forest': round(random.uniform(85, 95), 2),
            'logistic_regression': round(random.uniform(75, 85), 2),
            'svm': round(random.uniform(80, 90), 2)
        }

        accuracy = results.get(model, "Unknown Model")

        return JsonResponse({'result': f"{model.replace('_', ' ').title()} trained with {mc_runs} MC runs. Accuracy: {accuracy}%"})

    return JsonResponse({'error': 'Invalid request'}, status=400)