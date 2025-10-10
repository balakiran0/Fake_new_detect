import json
import pickle
from django.contrib.sessions.serializers import JSONSerializer
from django.core.signing import JSONSerializer as BaseJSONSerializer


class CustomSessionSerializer(JSONSerializer):
    """
    Custom session serializer that handles allauth objects that can't be JSON serialized
    """
    def dumps(self, obj):
        try:
            # Try JSON first
            return super().dumps(obj)
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(obj)
    
    def loads(self, data):
        try:
            # Try JSON first
            return super().loads(data)
        except (TypeError, ValueError, json.JSONDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
