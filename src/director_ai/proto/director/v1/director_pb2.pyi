from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Role(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROLE_UNSPECIFIED: _ClassVar[Role]
    ROLE_SYSTEM: _ClassVar[Role]
    ROLE_USER: _ClassVar[Role]
    ROLE_ASSISTANT: _ClassVar[Role]
    ROLE_TOOL: _ClassVar[Role]

class HaltReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HALT_REASON_UNSPECIFIED: _ClassVar[HaltReason]
    HALT_REASON_NONE: _ClassVar[HaltReason]
    HALT_REASON_COHERENCE_BELOW_THRESHOLD: _ClassVar[HaltReason]
    HALT_REASON_INJECTION_DETECTED: _ClassVar[HaltReason]
    HALT_REASON_POLICY_VIOLATION: _ClassVar[HaltReason]
    HALT_REASON_TOKEN_TIMEOUT: _ClassVar[HaltReason]
    HALT_REASON_TOTAL_TIMEOUT: _ClassVar[HaltReason]
    HALT_REASON_CALLBACK_TIMEOUT: _ClassVar[HaltReason]

class TenantTier(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TENANT_TIER_UNSPECIFIED: _ClassVar[TenantTier]
    TENANT_TIER_INDIE: _ClassVar[TenantTier]
    TENANT_TIER_PRO: _ClassVar[TenantTier]
    TENANT_TIER_PERPETUAL: _ClassVar[TenantTier]
    TENANT_TIER_ENTERPRISE_PILOT: _ClassVar[TenantTier]
    TENANT_TIER_ENTERPRISE: _ClassVar[TenantTier]
ROLE_UNSPECIFIED: Role
ROLE_SYSTEM: Role
ROLE_USER: Role
ROLE_ASSISTANT: Role
ROLE_TOOL: Role
HALT_REASON_UNSPECIFIED: HaltReason
HALT_REASON_NONE: HaltReason
HALT_REASON_COHERENCE_BELOW_THRESHOLD: HaltReason
HALT_REASON_INJECTION_DETECTED: HaltReason
HALT_REASON_POLICY_VIOLATION: HaltReason
HALT_REASON_TOKEN_TIMEOUT: HaltReason
HALT_REASON_TOTAL_TIMEOUT: HaltReason
HALT_REASON_CALLBACK_TIMEOUT: HaltReason
TENANT_TIER_UNSPECIFIED: TenantTier
TENANT_TIER_INDIE: TenantTier
TENANT_TIER_PRO: TenantTier
TENANT_TIER_PERPETUAL: TenantTier
TENANT_TIER_ENTERPRISE_PILOT: TenantTier
TENANT_TIER_ENTERPRISE: TenantTier

class ChatMessage(_message.Message):
    __slots__ = ("role", "content", "name")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    role: Role
    content: str
    name: str
    def __init__(self, role: _Optional[_Union[Role, str]] = ..., content: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class ChatCompletionRequest(_message.Message):
    __slots__ = ("model", "messages", "temperature", "max_tokens", "stream", "tenant_id", "request_id")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    model: str
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    temperature: float
    max_tokens: int
    stream: bool
    tenant_id: str
    request_id: str
    def __init__(self, model: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., temperature: _Optional[float] = ..., max_tokens: _Optional[int] = ..., stream: bool = ..., tenant_id: _Optional[str] = ..., request_id: _Optional[str] = ...) -> None: ...

class ChatChoice(_message.Message):
    __slots__ = ("index", "message", "delta_content", "finish_reason")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DELTA_CONTENT_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    index: int
    message: ChatMessage
    delta_content: str
    finish_reason: str
    def __init__(self, index: _Optional[int] = ..., message: _Optional[_Union[ChatMessage, _Mapping]] = ..., delta_content: _Optional[str] = ..., finish_reason: _Optional[str] = ...) -> None: ...

class TokenUsage(_message.Message):
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    def __init__(self, prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., total_tokens: _Optional[int] = ...) -> None: ...

class ChatCompletionResponse(_message.Message):
    __slots__ = ("id", "model", "created_unix", "choices", "usage", "coherence")
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CREATED_UNIX_FIELD_NUMBER: _ClassVar[int]
    CHOICES_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    COHERENCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    model: str
    created_unix: int
    choices: _containers.RepeatedCompositeFieldContainer[ChatChoice]
    usage: TokenUsage
    coherence: CoherenceVerdict
    def __init__(self, id: _Optional[str] = ..., model: _Optional[str] = ..., created_unix: _Optional[int] = ..., choices: _Optional[_Iterable[_Union[ChatChoice, _Mapping]]] = ..., usage: _Optional[_Union[TokenUsage, _Mapping]] = ..., coherence: _Optional[_Union[CoherenceVerdict, _Mapping]] = ...) -> None: ...

class GroundingSource(_message.Message):
    __slots__ = ("source_id", "similarity", "nli_support")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    NLI_SUPPORT_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    similarity: float
    nli_support: float
    def __init__(self, source_id: _Optional[str] = ..., similarity: _Optional[float] = ..., nli_support: _Optional[float] = ...) -> None: ...

class CoherenceVerdict(_message.Message):
    __slots__ = ("score", "halted", "halt_reason", "hard_limit", "score_lower", "score_upper", "sources", "message")
    SCORE_FIELD_NUMBER: _ClassVar[int]
    HALTED_FIELD_NUMBER: _ClassVar[int]
    HALT_REASON_FIELD_NUMBER: _ClassVar[int]
    HARD_LIMIT_FIELD_NUMBER: _ClassVar[int]
    SCORE_LOWER_FIELD_NUMBER: _ClassVar[int]
    SCORE_UPPER_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    score: float
    halted: bool
    halt_reason: HaltReason
    hard_limit: float
    score_lower: float
    score_upper: float
    sources: _containers.RepeatedCompositeFieldContainer[GroundingSource]
    message: str
    def __init__(self, score: _Optional[float] = ..., halted: bool = ..., halt_reason: _Optional[_Union[HaltReason, str]] = ..., hard_limit: _Optional[float] = ..., score_lower: _Optional[float] = ..., score_upper: _Optional[float] = ..., sources: _Optional[_Iterable[_Union[GroundingSource, _Mapping]]] = ..., message: _Optional[str] = ...) -> None: ...

class ScoreClaimRequest(_message.Message):
    __slots__ = ("claim", "documents", "tenant_id", "request_id", "threshold")
    CLAIM_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    claim: str
    documents: _containers.RepeatedScalarFieldContainer[str]
    tenant_id: str
    request_id: str
    threshold: float
    def __init__(self, claim: _Optional[str] = ..., documents: _Optional[_Iterable[str]] = ..., tenant_id: _Optional[str] = ..., request_id: _Optional[str] = ..., threshold: _Optional[float] = ...) -> None: ...

class ScoreClaimResponse(_message.Message):
    __slots__ = ("verdict", "latency_ms")
    VERDICT_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    verdict: CoherenceVerdict
    latency_ms: int
    def __init__(self, verdict: _Optional[_Union[CoherenceVerdict, _Mapping]] = ..., latency_ms: _Optional[int] = ...) -> None: ...

class ScoreTokenRequest(_message.Message):
    __slots__ = ("tenant_id", "request_id", "accumulated_text", "next_token", "documents")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ACCUMULATED_TEXT_FIELD_NUMBER: _ClassVar[int]
    NEXT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    request_id: str
    accumulated_text: str
    next_token: str
    documents: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, tenant_id: _Optional[str] = ..., request_id: _Optional[str] = ..., accumulated_text: _Optional[str] = ..., next_token: _Optional[str] = ..., documents: _Optional[_Iterable[str]] = ...) -> None: ...

class ScoreTokenResponse(_message.Message):
    __slots__ = ("verdict",)
    VERDICT_FIELD_NUMBER: _ClassVar[int]
    verdict: CoherenceVerdict
    def __init__(self, verdict: _Optional[_Union[CoherenceVerdict, _Mapping]] = ...) -> None: ...

class Tenant(_message.Message):
    __slots__ = ("tenant_id", "display_name", "tier", "created_unix", "rpm_limit", "rpd_limit", "api_key_fingerprints")
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    CREATED_UNIX_FIELD_NUMBER: _ClassVar[int]
    RPM_LIMIT_FIELD_NUMBER: _ClassVar[int]
    RPD_LIMIT_FIELD_NUMBER: _ClassVar[int]
    API_KEY_FINGERPRINTS_FIELD_NUMBER: _ClassVar[int]
    tenant_id: str
    display_name: str
    tier: TenantTier
    created_unix: int
    rpm_limit: int
    rpd_limit: int
    api_key_fingerprints: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, tenant_id: _Optional[str] = ..., display_name: _Optional[str] = ..., tier: _Optional[_Union[TenantTier, str]] = ..., created_unix: _Optional[int] = ..., rpm_limit: _Optional[int] = ..., rpd_limit: _Optional[int] = ..., api_key_fingerprints: _Optional[_Iterable[str]] = ...) -> None: ...

class APIKeyMetadata(_message.Message):
    __slots__ = ("fingerprint", "tenant_id", "issued_unix", "expires_unix", "revoked")
    FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    ISSUED_UNIX_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_UNIX_FIELD_NUMBER: _ClassVar[int]
    REVOKED_FIELD_NUMBER: _ClassVar[int]
    fingerprint: str
    tenant_id: str
    issued_unix: int
    expires_unix: int
    revoked: bool
    def __init__(self, fingerprint: _Optional[str] = ..., tenant_id: _Optional[str] = ..., issued_unix: _Optional[int] = ..., expires_unix: _Optional[int] = ..., revoked: bool = ...) -> None: ...

class AuditRecord(_message.Message):
    __slots__ = ("timestamp", "request_id", "tenant_id", "api_key_fingerprint", "query_hash", "response_length", "verdict", "policy_violations", "latency_ms", "model")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    API_KEY_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    QUERY_HASH_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    VERDICT_FIELD_NUMBER: _ClassVar[int]
    POLICY_VIOLATIONS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    timestamp: str
    request_id: str
    tenant_id: str
    api_key_fingerprint: str
    query_hash: str
    response_length: int
    verdict: CoherenceVerdict
    policy_violations: _containers.RepeatedScalarFieldContainer[str]
    latency_ms: int
    model: str
    def __init__(self, timestamp: _Optional[str] = ..., request_id: _Optional[str] = ..., tenant_id: _Optional[str] = ..., api_key_fingerprint: _Optional[str] = ..., query_hash: _Optional[str] = ..., response_length: _Optional[int] = ..., verdict: _Optional[_Union[CoherenceVerdict, _Mapping]] = ..., policy_violations: _Optional[_Iterable[str]] = ..., latency_ms: _Optional[int] = ..., model: _Optional[str] = ...) -> None: ...
