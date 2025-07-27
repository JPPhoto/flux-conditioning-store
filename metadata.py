from typing import Any, Dict

from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    ImageField,
    InputField,
    InvocationContext,
    OutputField,
    StringCollectionOutput,
    BooleanCollectionOutput,
    IntegerCollectionOutput,
    FloatCollectionOutput,
    invocation,
    invocation_output,
)
from invokeai.backend.util.logging import warning, error


class BaseExtractImageCollectionMetadataItemInvocation(BaseInvocation):
    """
    Base class for extracting specified metadata values from a collection of input images.
    It takes an image collection and a metadata key string input.
    For each image in the collection, it attempts to retrieve the value associated
    with the provided key. Subclasses will determine the output collection type.
    """

    images: list[ImageField] = InputField(
        description="A collection of images from which to extract metadata.",
        title="Image Collection",
        ui_order=0,
    )

    key: str = InputField(
        description="Metadata key to extract values for Output. Leave empty to ignore.",
        title="Metadata Key",
        default="",
        ui_order=1,
    )

    def _extract_and_process_metadata(
        self, context: InvocationContext
    ) -> list[Any]:
        """
        Helper method to extract metadata values, handling missing keys and errors.
        Returns a list of raw extracted values (or empty strings/None for missing/errors).
        """
        collected_raw_values: list[Any] = []

        for img_field in self.images:
            try:
                metadata: Dict[str, Any] = {}
                image_metadata = context.images.get_metadata(img_field.image_name)
                if image_metadata is not None:
                    metadata.update(image_metadata.root)

                if not metadata:
                    warning(
                        f"No metadata found for image: '{img_field.image_name}'. Appending empty value."
                    )
                    collected_raw_values.append(None)  # Use None to indicate no value
                    continue

                if self.key:
                    # Use .get() with a default of None to gracefully handle missing self.key
                    extracted_value = metadata.get(self.key, None)
                    collected_raw_values.append(extracted_value)
                else:
                    collected_raw_values.append(None)

            except Exception as e:
                error(f"Error processing image '{img_field.image_name}': {e}")
                collected_raw_values.append(None)

        return collected_raw_values

    # The invoke method will be implemented by subclasses to define the specific output type
    # and how to cast the collected raw values.
    def invoke(self, context: InvocationContext) -> BaseInvocationOutput:
        raise NotImplementedError("Subclasses must implement the invoke method.")


@invocation(
    "extract_image_collection_metadata_string",
    title="Extract Image Collection Metadata (String)",
    tags=["image", "metadata", "extraction", "collection", "utility", "string"],
    category="metadata",
    version="1.0.0",
)
class ExtractImageCollectionMetadataStringInvocation(
    BaseExtractImageCollectionMetadataItemInvocation
):
    """
    This node extracts specified metadata values as strings from a collection of input images.
    """

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        collected_raw_values = self._extract_and_process_metadata(context)
        # Convert all values to string. None becomes "None".
        processed_values = [str(v) if v is not None else "" for v in collected_raw_values]
        return StringCollectionOutput(collection=processed_values)


@invocation(
    "extract_image_collection_metadata_boolean",
    title="Extract Image Collection Metadata (Bool)",
    tags=["image", "metadata", "extraction", "collection", "utility", "boolean"],
    category="metadata",
    version="1.0.0",
)
class ExtractImageCollectionMetadataBooleanInvocation(
    BaseExtractImageCollectionMetadataItemInvocation
):
    """
    This node extracts specified metadata values as booleans from a collection of input images.
    Values are converted to boolean: truthy values become True, falsy values (including None, empty string, 0) become False.
    """

    def invoke(self, context: InvocationContext) -> BooleanCollectionOutput:
        collected_raw_values = self._extract_and_process_metadata(context)
        # Convert values to boolean. None, 0, empty string, etc. become False.
        processed_values = [bool(v) for v in collected_raw_values]
        return BooleanCollectionOutput(collection=processed_values)


@invocation(
    "extract_image_collection_metadata_integer",
    title="Extract Image Collection Metadata (Int)",
    tags=["image", "metadata", "extraction", "collection", "utility", "integer"],
    category="metadata",
    version="1.0.0",
)
class ExtractImageCollectionMetadataIntegerInvocation(
    BaseExtractImageCollectionMetadataItemInvocation
):
    """
    This node extracts specified metadata values as integers from a collection of input images.
    Non-integer values will attempt to be converted. If conversion fails, 0 is used.
    """

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        collected_raw_values = self._extract_and_process_metadata(context)
        processed_values: list[int] = []
        for v in collected_raw_values:
            try:
                if v is None:
                    processed_values.append(0)  # Default for None
                else:
                    processed_values.append(int(v))
            except (ValueError, TypeError):
                warning(f"Could not convert '{v}' to integer. Using 0.")
                processed_values.append(0)
        return IntegerCollectionOutput(collection=processed_values)


@invocation(
    "extract_image_collection_metadata_float",
    title="Extract Image Collection Metadata (Float)",
    tags=["image", "metadata", "extraction", "collection", "utility", "float"],
    category="metadata",
    version="1.0.0",
)
class ExtractImageCollectionMetadataFloatInvocation(
    BaseExtractImageCollectionMetadataItemInvocation
):
    """
    This node extracts specified metadata values as floats from a collection of input images.
    Non-float values will attempt to be converted. If conversion fails, 0.0 is used.
    """

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        collected_raw_values = self._extract_and_process_metadata(context)
        processed_values: list[float] = []
        for v in collected_raw_values:
            try:
                if v is None:
                    processed_values.append(0.0)  # Default for None
                else:
                    processed_values.append(float(v))
            except (ValueError, TypeError):
                warning(f"Could not convert '{v}' to float. Using 0.0.")
                processed_values.append(0.0)
        return FloatCollectionOutput(collection=processed_values)
