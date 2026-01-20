from __future__ import annotations

from django.urls import path

from .views import (
    CollectionDetailView,
    CollectionsView,
    DocumentChunksView,
    HealthView,
    IngestFileView,
    IngestTextView,
    ReconstructView,
    SearchView,
)

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("collections/", CollectionsView.as_view(), name="collections"),
    path("collections/<str:collection_name>/", CollectionDetailView.as_view(), name="collection-detail"),
    path("ingest/text/", IngestTextView.as_view(), name="ingest-text"),
    path("ingest/file/", IngestFileView.as_view(), name="ingest-file"),
    path("search/", SearchView.as_view(), name="search"),
    path("documents/chunks/", DocumentChunksView.as_view(), name="document-chunks"),
    path("documents/reconstruct/", ReconstructView.as_view(), name="document-reconstruct"),
]

