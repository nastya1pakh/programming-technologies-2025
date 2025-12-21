from django.urls import path
from .views import SearchView

urlpatterns = [
    path('milvus/', SearchView.as_view(), name='search'),
]
