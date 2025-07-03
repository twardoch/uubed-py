"""
Custom exception hierarchy for uubed package.

Provides detailed error messages with user guidance and error codes for programmatic handling.
"""

from typing import Optional, Any, Dict, List


class UubedError(Exception):
    """Base exception for all uubed-related errors."""
    
    def __init__(
        self, 
        message: str, 
        suggestion: Optional[str] = None, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a UubedError.
        
        Args:
            message: Human-readable error description
            suggestion: Helpful suggestion for fixing the error
            error_code: Machine-readable error code for programmatic handling
            context: Additional context information for debugging
        """
        self.suggestion = suggestion
        self.error_code = error_code
        self.context = context or {}
        
        # Format the complete error message
        full_message = message
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"
        if error_code:
            full_message += f"\nError Code: {error_code}"
            
        super().__init__(full_message)


class UubedValidationError(UubedError):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        parameter: Optional[str] = None,
        expected: Optional[str] = None,
        received: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize a validation error.
        
        Args:
            message: Base error message
            parameter: Name of the parameter that failed validation
            expected: Description of expected value/type
            received: Description of received value/type
            suggestion: Helpful suggestion for fixing the error
        """
        context = {}
        if parameter:
            context['parameter'] = parameter
        if expected:
            context['expected'] = expected
        if received:
            context['received'] = received
            
        # Enhanced message with parameter details
        if parameter and expected and received:
            enhanced_message = f"{message}\nParameter '{parameter}': expected {expected}, got {received}"
        else:
            enhanced_message = message
            
        super().__init__(
            enhanced_message,
            suggestion=suggestion,
            error_code="VALIDATION_ERROR",
            context=context
        )


class UubedEncodingError(UubedError):
    """Raised when encoding operations fail."""
    
    def __init__(
        self, 
        message: str, 
        method: Optional[str] = None,
        embedding_shape: Optional[tuple] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize an encoding error.
        
        Args:
            message: Base error message
            method: Encoding method that failed
            embedding_shape: Shape of the embedding that caused the error
            suggestion: Helpful suggestion for fixing the error
        """
        context = {}
        if method:
            context['method'] = method
        if embedding_shape:
            context['embedding_shape'] = embedding_shape
            
        # Enhanced message with method details
        if method:
            enhanced_message = f"Encoding failed with method '{method}': {message}"
        else:
            enhanced_message = f"Encoding failed: {message}"
            
        super().__init__(
            enhanced_message,
            suggestion=suggestion,
            error_code="ENCODING_ERROR",
            context=context
        )


class UubedDecodingError(UubedError):
    """Raised when decoding operations fail."""
    
    def __init__(
        self, 
        message: str, 
        method: Optional[str] = None,
        encoded_string: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize a decoding error.
        
        Args:
            message: Base error message
            method: Decoding method that failed
            encoded_string: The encoded string that caused the error (truncated if long)
            suggestion: Helpful suggestion for fixing the error
        """
        context = {}
        if method:
            context['method'] = method
        if encoded_string:
            # Truncate long strings for readability
            if len(encoded_string) > 100:
                context['encoded_string'] = encoded_string[:100] + "..."
            else:
                context['encoded_string'] = encoded_string
                
        # Enhanced message with method details
        if method:
            enhanced_message = f"Decoding failed with method '{method}': {message}"
        else:
            enhanced_message = f"Decoding failed: {message}"
            
        super().__init__(
            enhanced_message,
            suggestion=suggestion,
            error_code="DECODING_ERROR",
            context=context
        )


class UubedResourceError(UubedError):
    """Raised when resource management fails (memory, files, GPU, etc.)."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        available: Optional[str] = None,
        required: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize a resource error.
        
        Args:
            message: Base error message
            resource_type: Type of resource (memory, disk, GPU, etc.)
            available: Description of available resource
            required: Description of required resource
            suggestion: Helpful suggestion for fixing the error
        """
        context = {}
        if resource_type:
            context['resource_type'] = resource_type
        if available:
            context['available'] = available
        if required:
            context['required'] = required
            
        # Enhanced message with resource details
        if resource_type and available and required:
            enhanced_message = f"{message}\n{resource_type}: required {required}, available {available}"
        else:
            enhanced_message = message
            
        super().__init__(
            enhanced_message,
            suggestion=suggestion,
            error_code="RESOURCE_ERROR",
            context=context
        )


class UubedConnectionError(UubedError):
    """Raised when external service connections fail."""
    
    def __init__(
        self, 
        message: str, 
        service: Optional[str] = None,
        operation: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize a connection error.
        
        Args:
            message: Base error message
            service: Name of the external service
            operation: Operation that was being performed
            suggestion: Helpful suggestion for fixing the error
        """
        context = {}
        if service:
            context['service'] = service
        if operation:
            context['operation'] = operation
            
        # Enhanced message with service details
        if service and operation:
            enhanced_message = f"Connection to {service} failed during {operation}: {message}"
        elif service:
            enhanced_message = f"Connection to {service} failed: {message}"
        else:
            enhanced_message = f"Connection failed: {message}"
            
        super().__init__(
            enhanced_message,
            suggestion=suggestion,
            error_code="CONNECTION_ERROR",
            context=context
        )


class UubedConfigurationError(UubedError):
    """Raised when configuration or setup issues are detected."""
    
    def __init__(
        self, 
        message: str, 
        component: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize a configuration error.
        
        Args:
            message: Base error message
            component: Component or feature that is misconfigured
            suggestion: Helpful suggestion for fixing the error
        """
        context = {}
        if component:
            context['component'] = component
            
        # Enhanced message with component details
        if component:
            enhanced_message = f"Configuration error in {component}: {message}"
        else:
            enhanced_message = f"Configuration error: {message}"
            
        super().__init__(
            enhanced_message,
            suggestion=suggestion,
            error_code="CONFIGURATION_ERROR",
            context=context
        )


# Convenience functions for common error patterns

def validation_error(message: str, parameter: str, expected: str, received: str) -> UubedValidationError:
    """Create a validation error with standard formatting."""
    suggestion_map = {
        'type': "Check the input type and convert if necessary",
        'range': "Ensure values are within the valid range",
        'shape': "Check the input dimensions and reshape if needed",
        'method': "Use one of the supported encoding methods: eq64, shq64, t8q64, zoq64"
    }
    
    suggestion = None
    for key, value in suggestion_map.items():
        if key in expected.lower() or key in message.lower():
            suggestion = value
            break
    
    return UubedValidationError(
        message=message,
        parameter=parameter,
        expected=expected,
        received=received,
        suggestion=suggestion
    )


def encoding_error(message: str, method: str, suggestion: Optional[str] = None) -> UubedEncodingError:
    """Create an encoding error with method-specific suggestions."""
    method_suggestions = {
        'eq64': "Ensure embedding values are in range 0-255 and dimensions are valid",
        'shq64': "Check the 'planes' parameter (must be multiple of 8, typically 64-256)",
        't8q64': "Check the 'k' parameter (must be positive, typically 8-32)",
        'zoq64': "Ensure embedding has valid dimensions (typically powers of 2)"
    }
    
    if not suggestion and method in method_suggestions:
        suggestion = method_suggestions[method]
    
    return UubedEncodingError(
        message=message,
        method=method,
        suggestion=suggestion
    )


def resource_error(message: str, resource_type: str, suggestion: Optional[str] = None) -> UubedResourceError:
    """Create a resource error with type-specific suggestions."""
    resource_suggestions = {
        'memory': "Try reducing batch_size or using streaming operations",
        'gpu': "Check CUDA installation or use CPU fallback",
        'disk': "Check available disk space and file permissions",
        'network': "Check internet connection and service availability"
    }
    
    if not suggestion and resource_type.lower() in resource_suggestions:
        suggestion = resource_suggestions[resource_type.lower()]
    
    return UubedResourceError(
        message=message,
        resource_type=resource_type,
        suggestion=suggestion
    )


def connection_error(message: str, service: str, suggestion: Optional[str] = None) -> UubedConnectionError:
    """Create a connection error with service-specific suggestions."""
    service_suggestions = {
        'pinecone': "Check your API key and environment settings",
        'weaviate': "Verify the Weaviate server URL and authentication",
        'qdrant': "Check the Qdrant server URL and API key",
        'chromadb': "Ensure ChromaDB is properly installed and configured"
    }
    
    if not suggestion and service.lower() in service_suggestions:
        suggestion = service_suggestions[service.lower()]
    
    return UubedConnectionError(
        message=message,
        service=service,
        suggestion=suggestion
    )


def configuration_error(
    message: str, 
    config_file: Optional[str] = None, 
    suggestion: Optional[str] = None
) -> UubedConfigurationError:
    """Create a configuration error with file-specific suggestions."""
    config_suggestions = {
        'json': "Check JSON syntax: quotes around strings, proper comma placement",
        'toml': "Check TOML syntax: quotes around strings, section headers in brackets",
        'missing': "Create a configuration file or check the file path",
        'permissions': "Check file permissions and directory access"
    }
    
    if not suggestion:
        if config_file:
            if '.json' in config_file.lower():
                suggestion = config_suggestions['json']
            elif '.toml' in config_file.lower():
                suggestion = config_suggestions['toml']
            else:
                suggestion = "Check configuration file syntax and format"
        else:
            suggestion = config_suggestions['missing']
    
    return UubedConfigurationError(
        message=message,
        config_file=config_file,
        suggestion=suggestion
    )