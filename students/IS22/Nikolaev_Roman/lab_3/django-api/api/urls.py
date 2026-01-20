# api/urls.py
from django.urls import path

from .views import CollectionInfoAPIView, DocumentReconstructAPIView, SearchAPIView

urlpatterns = [
    path("search/", SearchAPIView.as_view(), name="search"),
    path("collections/<str:name>/info/", CollectionInfoAPIView.as_view(), name='collections-info'),
    path('reconstruct/', DocumentReconstructAPIView.as_view(), name='reconstruct-doc'),
]
