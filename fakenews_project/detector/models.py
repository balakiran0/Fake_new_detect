# detector/models.py
from django.db import models

class AnalysisResult(models.Model):
    input_text = models.TextField()
    is_fake = models.BooleanField()
    confidence_score = models.FloatField()
    generated_notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for: {self.input_text[:50]}..."
    