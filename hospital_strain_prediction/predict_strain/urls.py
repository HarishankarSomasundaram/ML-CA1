from django.urls import path
from . import views
from .views import train_model

urlpatterns = [
    path('', views.predict_strain, name='predict_strain'),
    path('train-model/', train_model, name='train_model'),
]