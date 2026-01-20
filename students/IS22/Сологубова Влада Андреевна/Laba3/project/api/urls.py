from django.urls import path
from .views import (
    CreateCollectionView,
    UploadDocumentsView,
    SemanticSearchView,
    CollectionInfoView,
    DocumentChunksView
)

urlpatterns = [
    path("collections/create/", CreateCollectionView.as_view()),
    path("collections/<str:name>/info/", CollectionInfoView.as_view()),
    path("documents/upload/", UploadDocumentsView.as_view()),
    path("documents/<str:collection_name>/chunks/<path:file_path>/",
         DocumentChunksView.as_view()),
    path("search/", SemanticSearchView.as_view()),
]
