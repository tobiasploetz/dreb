import { MODELS } from "./models.generated.js";
import {
	type Api,
	type KnownProvider,
	type Model,
	THINKING_ENCODING_KINDS,
	THINKING_OFF_MODES,
	THINKING_TIERS,
	type ThinkingCapability,
	type ThinkingCapabilityDeclaration,
	type ThinkingEncodingKind,
	type ThinkingLevel,
	type ThinkingOffMode,
	type Usage,
} from "./types.js";

const modelRegistry: Map<string, Map<string, Model<Api>>> = new Map();

// Initialize registry from MODELS on module load
for (const [provider, models] of Object.entries(MODELS)) {
	const providerModels = new Map<string, Model<Api>>();
	for (const [id, model] of Object.entries(models)) {
		providerModels.set(id, model as Model<Api>);
	}
	modelRegistry.set(provider, providerModels);
}

type ModelApi<
	TProvider extends KnownProvider,
	TModelId extends keyof (typeof MODELS)[TProvider],
> = (typeof MODELS)[TProvider][TModelId] extends { api: infer TApi } ? (TApi extends Api ? TApi : never) : never;

export function getModel<TProvider extends KnownProvider, TModelId extends keyof (typeof MODELS)[TProvider]>(
	provider: TProvider,
	modelId: TModelId,
): Model<ModelApi<TProvider, TModelId>> {
	const providerModels = modelRegistry.get(provider);
	return providerModels?.get(modelId as string) as Model<ModelApi<TProvider, TModelId>>;
}

/**
 * Check if a model ID looks like an alias (no date suffix).
 * Aliases are preferred over dated versions when fuzzy matching.
 *
 * IDs ending with `-latest` are treated as aliases.
 * IDs ending with a date pattern (`-YYYYMMDD`) are treated as dated versions.
 */
export function isModelAlias(id: string): boolean {
	if (id.endsWith("-latest")) return true;
	return !/-\d{8}$/.test(id);
}

/**
 * Find a model by fuzzy matching against the provider's registered models.
 *
 * Resolution order:
 * 1. Exact match by provider + model ID (via registry Map.get)
 * 2. Case-insensitive substring match against model ID and display name
 * 3. Among matches, prefer aliases (non-dated IDs) over dated versions
 * 4. Among ties, pick the lexicographically highest (latest) ID
 *
 * This is the same matching logic used by the CLI, subagent model resolution,
 * and interactive mode — centralised here so tests can exercise the real path.
 *
 * @example
 * findModel("anthropic", "sonnet")  // → latest claude-sonnet alias
 * findModel("anthropic", "haiku")   // → latest claude-haiku alias
 * findModel("openai", "gpt-5")     // → latest gpt-5 alias
 */
export function findModel(provider: string, pattern: string): Model<Api> | undefined {
	const providerModels = modelRegistry.get(provider);
	if (!providerModels) return undefined;

	// Try exact match first
	const exact = providerModels.get(pattern);
	if (exact) return exact;

	// Substring match (case-insensitive)
	const normalizedPattern = pattern.toLowerCase();
	const matches = Array.from(providerModels.values()).filter(
		(m) => m.id.toLowerCase().includes(normalizedPattern) || m.name?.toLowerCase().includes(normalizedPattern),
	);

	if (matches.length === 0) return undefined;
	if (matches.length === 1) return matches[0];

	// Multiple matches — separate into aliases and dated versions
	const aliases = matches.filter((m) => isModelAlias(m.id));
	const datedVersions = matches.filter((m) => !isModelAlias(m.id));

	if (aliases.length > 0) {
		// Prefer alias — if multiple, pick the lexicographically highest
		aliases.sort((a, b) => b.id.localeCompare(a.id));
		return aliases[0];
	}

	// All dated — prefer the latest
	datedVersions.sort((a, b) => b.id.localeCompare(a.id));
	return datedVersions[0];
}

/**
 * Find a model by fuzzy matching against a flat array of models.
 * Same algorithm as findModel() but operates on an arbitrary model list
 * instead of the built-in registry.
 *
 * Used by model-resolver.ts and other code that manages its own model lists.
 */
export function findModelInList(pattern: string, models: Model<Api>[]): Model<Api> | undefined {
	if (models.length === 0) return undefined;

	const normalizedPattern = pattern.toLowerCase();

	// Exact ID match (case-insensitive)
	const exactById = models.find((m) => m.id.toLowerCase() === normalizedPattern);
	if (exactById) return exactById;

	// Substring match (case-insensitive)
	const matches = models.filter(
		(m) => m.id.toLowerCase().includes(normalizedPattern) || m.name?.toLowerCase().includes(normalizedPattern),
	);

	if (matches.length === 0) return undefined;
	if (matches.length === 1) return matches[0];

	// Multiple matches — separate into aliases and dated versions
	const aliases = matches.filter((m) => isModelAlias(m.id));
	const datedVersions = matches.filter((m) => !isModelAlias(m.id));

	if (aliases.length > 0) {
		aliases.sort((a, b) => b.id.localeCompare(a.id));
		return aliases[0];
	}

	datedVersions.sort((a, b) => b.id.localeCompare(a.id));
	return datedVersions[0];
}

export function getProviders(): KnownProvider[] {
	return Array.from(modelRegistry.keys()) as KnownProvider[];
}

export function getModels<TProvider extends KnownProvider>(
	provider: TProvider,
): Model<ModelApi<TProvider, keyof (typeof MODELS)[TProvider]>>[] {
	const models = modelRegistry.get(provider);
	return models ? (Array.from(models.values()) as Model<ModelApi<TProvider, keyof (typeof MODELS)[TProvider]>>[]) : [];
}

export function calculateCost<TApi extends Api>(model: Model<TApi>, usage: Usage): Usage["cost"] {
	usage.cost.input = (model.cost.input / 1000000) * usage.input;
	usage.cost.output = (model.cost.output / 1000000) * usage.output;
	usage.cost.cacheRead = (model.cost.cacheRead / 1000000) * usage.cacheRead;
	usage.cost.cacheWrite = (model.cost.cacheWrite / 1000000) * usage.cacheWrite;
	usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite;
	return usage.cost;
}

interface ClaudeFamilyVersion {
	family: "opus" | "sonnet" | "haiku";
	major: number;
	minor?: number;
}

/** Parse Claude family versions without mistaking long date suffixes for versions. */
function claudeFamilyVersion(modelId: string): ClaudeFamilyVersion | undefined {
	const match = modelId.match(/(opus|sonnet|haiku)-(\d{1,2})(?:[.-](\d{1,2}))?(?!\d)/);
	if (!match) return undefined;
	return {
		family: match[1] as ClaudeFamilyVersion["family"],
		major: Number.parseInt(match[2], 10),
		minor: match[3] === undefined ? undefined : Number.parseInt(match[3], 10),
	};
}

/** True when a parsed Claude version is at least `major.minor`. */
function claudeAtLeast(claude: ClaudeFamilyVersion, major: number, minor = 0): boolean {
	return claude.major > major || (claude.major === major && (claude.minor ?? 0) >= minor);
}

/**
 * Anthropic-native transports, where dreb controls the Anthropic `thinking` /
 * `output_config.effort` request fields directly: the Anthropic Messages API
 * (Anthropic, GitHub Copilot's `/v1/messages`, opencode, custom Anthropic-compatible
 * endpoints) and Amazon Bedrock. Claude served through OpenAI-compatible APIs
 * (OpenRouter, Copilot Fable, ...) keeps the legacy capability rules because those
 * transports map effort themselves.
 */
function isAnthropicNativeTransport<TApi extends Api>(model: Model<TApi>): boolean {
	return model.api === "anthropic-messages" || model.api === "bedrock-converse-stream";
}

/** Claude model on an Anthropic-native transport, or undefined. */
function nativeClaude<TApi extends Api>(model: Model<TApi>): ClaudeFamilyVersion | undefined {
	return isAnthropicNativeTransport(model) ? claudeFamilyVersion(model.id) : undefined;
}

/**
 * Parse Qwen family versions without mistaking parameter counts for minor versions.
 * Only a dot separates major from minor (qwen3.8-27b, Qwen-3.8); a hyphen after the
 * major version introduces the parameter size (qwen3-32b → 3.0, not 3.32). A capture
 * directly followed by a `b`-suffixed size token is a parameter count, not a version
 * (qwen-32b, Qwen-7B-Chat, deepseek-r1-distill-qwen-32b), and Qwen majors never reach
 * two digits, so only a single digit is accepted.
 */
function qwenFamilyVersion(modelId: string): { major: number; minor: number } | undefined {
	const match = modelId.match(/\bqwen[\s_-]*v?(\d)(?:\.(\d+))?(?![\d.]*b\b)/i);
	if (!match) return undefined;
	return {
		major: Number.parseInt(match[1], 10),
		minor: match[2] === undefined ? 0 : Number.parseInt(match[2], 10),
	};
}

/**
 * Check whether a model id belongs to Qwen 3.8 or a later Qwen generation — the first
 * Qwen family with graded `reasoning_effort` tiers (low/medium/xhigh, default xhigh).
 */
export function isQwen38OrLater(modelId: string): boolean {
	const qwen = qwenFamilyVersion(modelId);
	if (!qwen) return false;
	return qwen.major > 3 || (qwen.major === 3 && qwen.minor >= 8);
}

/**
 * Legacy heuristic: does this model id belong to a family with a native `max` tier?
 *
 * Supported today:
 * - GPT-5.6 model families (Sol, Terra, Luna, and the alias)
 * - GPT-6 model families (Astra and future variants)
 * - On Anthropic-native transports (Anthropic Messages, including GitHub Copilot, and
 *   Bedrock): Claude Opus/Sonnet 4.6+ (including the Claude 5 families) and Haiku 5.5+
 *
 * This is the inference fallback used when a model carries no declared
 * {@link ThinkingCapability}. Prefer `getThinkingCapability(model)`.
 */
function infersMax<TApi extends Api>(model: Model<TApi>): boolean {
	const id = model.id.toLowerCase();
	// Anchored to the id start or a gateway "/" prefix so unrelated ids
	// (e.g. "gpt-60") do not match.
	if (/(?:^|\/)gpt-5\.6(?:$|[-.])/.test(id) || /(?:^|\/)gpt-6(?:$|[-.])/.test(id)) return true;

	const claude = nativeClaude(model);
	if (!claude) return false;
	if (claude.family === "haiku") return claudeAtLeast(claude, 5, 5);
	return claudeAtLeast(claude, 4, 6);
}

/**
 * Legacy heuristic: does this model id belong to a family with a native `xhigh` tier?
 *
 * Supported today:
 * - GPT-5.2 through GPT-5.6 model families
 * - GPT-6 model families (Astra and future variants)
 * - Claude on Anthropic-native transports: Opus 4.7+, Sonnet 5+, Haiku 5.5+
 *   (Opus 4.6 and Sonnet 4.6 have `max` but no native `xhigh`)
 * - Claude on OpenAI-compatible transports (OpenRouter, ...): Opus 4.6+ and Sonnet 5+
 *   (the legacy rule; those transports map effort themselves)
 * - Kimi Code K3 (xhigh maps to its advertised "max" effort)
 * - Qwen 3.8+ model families (xhigh is their top native effort tier)
 */
function infersXhigh<TApi extends Api>(model: Model<TApi>): boolean {
	if (model.provider === "kimi-coding-oauth" && model.id === "k3") return true;
	if (isQwen38OrLater(model.id)) return true;
	if (/gpt-(?:5\.[2-6]|6)/.test(model.id)) return true;

	const claude = claudeFamilyVersion(model.id);
	if (!claude) return false;
	if (isAnthropicNativeTransport(model)) {
		if (claude.family === "opus") return claudeAtLeast(claude, 4, 7);
		if (claude.family === "sonnet") return claudeAtLeast(claude, 5);
		return claudeAtLeast(claude, 5, 5);
	}
	// Legacy rule for proxied Claude transports (unchanged; Haiku never qualified).
	if (claude.family === "opus") return claudeAtLeast(claude, 4, 6);
	if (claude.family === "sonnet") return claudeAtLeast(claude, 5);
	return false;
}

/**
 * Legacy heuristic: does this model use adaptive thinking (Opus/Sonnet 4.6+, Haiku 5+,
 * including the Claude 5 families), where `output_config.effort` and the
 * `thinkingDisplay` option are honored?
 *
 * Adaptive thinking is a property of the Claude family, not of the transport, so this
 * deliberately ignores `model.api` (a Claude id behind an OpenAI-compatible proxy still
 * answers true; those providers ignore the display option).
 */
function infersAdaptive<TApi extends Api>(model: Model<TApi>): boolean {
	const claude = claudeFamilyVersion(model.id);
	if (!claude) return false;
	if (claude.family === "haiku") return claudeAtLeast(claude, 5);
	return claudeAtLeast(claude, 4, 6);
}

/**
 * Legacy heuristic: the request shape that turns thinking off on an Anthropic-native
 * transport:
 * - `"omit"`: send neither a `thinking` field nor an effort (Opus 5.5+, where thinking
 *   cannot be disabled; the model default effort applies)
 * - `"between_tools"`: `thinking: {type: "between_tools"}` (Sonnet 5.5+, where
 *   `disabled` is rejected)
 * - `"disabled"`: `thinking: {type: "disabled"}` (every other model, and every model on
 *   non-native transports)
 */
function infersOffMode<TApi extends Api>(model: Model<TApi>): ThinkingOffMode {
	const claude = nativeClaude(model);
	if (!claude) return "disabled";
	if (claude.family === "opus" && claudeAtLeast(claude, 5, 5)) return "omit";
	if (claude.family === "sonnet" && claudeAtLeast(claude, 5, 5)) return "between_tools";
	return "disabled";
}

/**
 * Legacy heuristic: the wire shape this model's API uses for thinking.
 *
 * Adaptive Claude gets its own encoding; everything else is approximated from the API
 * (Anthropic Messages and Bedrock are budget-based, the OpenAI-shaped and Google APIs
 * take graded effort). Providers adopt this encoding in a follow-up; today only
 * `"adaptive"` is consumed, by {@link supportsAdaptiveThinking}.
 */
function infersEncodingKind<TApi extends Api>(model: Model<TApi>): ThinkingEncodingKind {
	if (infersAdaptive(model)) return "adaptive";
	if (!model.reasoning) return "toggle";
	const api = model.api ?? "";
	if (isAnthropicNativeTransport(model) || api.startsWith("google")) return "budget";
	return "effort";
}

/**
 * Infer thinking capability from a model id and API, with no declared data.
 *
 * This reproduces the behaviour of the per-family predicates exactly, so it is the
 * compatibility path for models that carry no `thinking` declaration.
 */
export function inferThinkingCapability<TApi extends Api>(model: Model<TApi> | undefined): ThinkingCapability {
	if (!model?.reasoning) {
		return {
			levels: [],
			off: model ? infersOffMode(model) : "disabled",
			// Adaptive is a property of the Claude family, not of the reasoning flag, so a
			// model object carrying only an id still reports its family's encoding.
			encoding: { kind: model ? infersEncodingKind(model) : "toggle" },
		};
	}

	const levels: ThinkingLevel[] = ["minimal", "low", "medium", "high"];
	if (infersXhigh(model)) levels.push("xhigh");
	if (infersMax(model)) levels.push("max");

	return {
		levels,
		off: infersOffMode(model),
		encoding: { kind: infersEncodingKind(model) },
	};
}

/**
 * Normalise a declared capability against the inferred one.
 *
 * Missing parts fall back to inference; tiers are filtered to the known scale, deduped
 * and sorted ascending; unknown `off`/`kind` values fall back to the inferred value.
 * This keeps SDK-constructed models and hand-written `models.json` entries safe even
 * when they bypass schema validation.
 */
function normalizeThinkingCapability<TApi extends Api>(
	model: Model<TApi> | undefined,
	declared: ThinkingCapabilityDeclaration | undefined,
): ThinkingCapability {
	const inferred = inferThinkingCapability(model);
	if (!declared) return inferred;

	const declaredLevels = declared.levels;
	const levels = Array.isArray(declaredLevels)
		? THINKING_TIERS.filter((tier) => declaredLevels.includes(tier))
		: inferred.levels;

	const off = declared.off && THINKING_OFF_MODES.includes(declared.off) ? declared.off : inferred.off;

	const declaredEncoding = declared.encoding ?? {};
	const kind =
		declaredEncoding.kind && THINKING_ENCODING_KINDS.includes(declaredEncoding.kind)
			? declaredEncoding.kind
			: inferred.encoding.kind;

	return { levels, off, encoding: { ...declaredEncoding, kind } };
}

/**
 * Resolve a model's thinking capability: declared data wins, otherwise inference.
 *
 * This is the single source of truth for "which thinking levels does this model have,
 * and how are they expressed". Every capability helper below derives from it.
 */
export function getThinkingCapability<TApi extends Api>(model: Model<TApi> | undefined): ThinkingCapability {
	return normalizeThinkingCapability(model, model?.thinking);
}

/**
 * Check if a model supports the native `max` reasoning tier.
 *
 * True when `max` is among the model's resolved {@link ThinkingCapability.levels}, i.e.
 * for the families in {@link infersMax} and for any model that declares it in
 * `models.json`.
 */
export function supportsMax<TApi extends Api>(model: Model<TApi>): boolean {
	return getThinkingCapability(model).levels.includes("max");
}

/**
 * Check if a model supports xhigh thinking level.
 *
 * True when `xhigh` is among the model's resolved {@link ThinkingCapability.levels}, i.e.
 * for the families in {@link infersXhigh} and for any model that declares it.
 */
export function supportsXhigh<TApi extends Api>(model: Model<TApi>): boolean {
	return getThinkingCapability(model).levels.includes("xhigh");
}

/**
 * Check if a model uses adaptive thinking (Opus/Sonnet 4.6+, Haiku 5+, including the
 * Claude 5 families), where `output_config.effort` and the `thinkingDisplay` option
 * are honored. Mirrors the per-provider internal checks.
 */
export function supportsAdaptiveThinking<TApi extends Api>(model: Model<TApi>): boolean {
	return getThinkingCapability(model).encoding.kind === "adaptive";
}

/**
 * Check whether thinking can be turned off for a model (dreb level `off`).
 *
 * False only for Claude Opus 5.5+ on Anthropic-native transports, where adaptive
 * thinking is always on and `thinking: {type: "disabled"}` is rejected. Callers that
 * offer selectable levels should hide `off` for such models; {@link clampThinkingLevel}
 * maps a requested `off` to `minimal`.
 */
export function supportsThinkingOff<TApi extends Api>(model: Model<TApi>): boolean {
	return getThinkingCapability(model).off !== "omit";
}

/** @deprecated Use {@link ThinkingOffMode} from `./types.js`. */
export type ClaudeThinkingOffMode = ThinkingOffMode;

/**
 * Select the request shape that turns thinking off for a model.
 *
 * @see {@link ThinkingOffMode}
 */
export function getClaudeThinkingOffMode<TApi extends Api>(model: Model<TApi>): ThinkingOffMode {
	return getThinkingCapability(model).off;
}

/**
 * Check whether a Claude model thinks by default when a request omits the `thinking`
 * field (the Claude 5 generation). Transports that otherwise omit the field to turn
 * thinking off (Bedrock) must send an explicit off config for these models.
 */
export function claudeThinksByDefault<TApi extends Api>(model: Model<TApi>): boolean {
	const claude = claudeFamilyVersion(model.id);
	return claude !== undefined && claude.major >= 5;
}

/** Every dreb thinking level, in ascending order. */
const THINKING_LEVEL_ORDER: readonly (ThinkingLevel | "off")[] = ["off", ...THINKING_TIERS];

/**
 * Thinking levels a model supports, in ascending order: `["off"]` for non-reasoning
 * models; otherwise `"off"` prepended when thinking can be turned off, followed by the
 * model's resolved tiers (which already omit `xhigh`/`max` where unsupported).
 */
export function getAvailableThinkingLevels<TApi extends Api>(
	model: Model<TApi> | undefined,
): (ThinkingLevel | "off")[] {
	// Non-reasoning models offer only `off`, regardless of their declared off mode.
	if (!model?.reasoning) return ["off"];

	const capability = getThinkingCapability(model);
	// A model with no tiers but disableable thinking still offers `off` alone.
	if (capability.off === "omit" && capability.levels.length > 0) return [...capability.levels];
	return ["off", ...capability.levels];
}

/**
 * Clamp a requested thinking level to what a model supports. This is the single
 * source of truth for thinking-level fallback across dreb: the nearest available
 * level at or above the request, else the highest available level. So:
 * - non-reasoning models → `off`
 * - `off` on models where thinking cannot be turned off → `minimal`
 * - unsupported `xhigh` → `max` when the model has `max` (Opus/Sonnet 4.6), else `high`
 * - unsupported `max` → `xhigh` when supported, else `high`
 * - an unknown level → the lowest available level
 */
export function clampThinkingLevel<TApi extends Api>(
	model: Model<TApi> | undefined,
	level: ThinkingLevel | "off",
): ThinkingLevel | "off" {
	const available = getAvailableThinkingLevels(model);
	const requested = THINKING_LEVEL_ORDER.indexOf(level);
	return (
		available.find((candidate) => THINKING_LEVEL_ORDER.indexOf(candidate) >= requested) ??
		available[available.length - 1]
	);
}

/**
 * Anthropic adaptive-thinking effort (`output_config.effort`) for a dreb thinking level,
 * after clamping it to the model. `minimal` → `low`; `xhigh` and `max` are sent as
 * themselves; an unset level (or `off` on a non-reasoning model) → `high`.
 */
export function getClaudeEffort<TApi extends Api>(
	model: Model<TApi>,
	level: ThinkingLevel | undefined,
): Exclude<ThinkingLevel, "minimal"> {
	const clamped = level === undefined ? "high" : clampThinkingLevel(model, level);
	return clamped === "minimal" ? "low" : clamped === "off" ? "high" : clamped;
}

/**
 * Check if two models are equal by comparing both their id and provider.
 * Returns false if either model is null or undefined.
 */
export function modelsAreEqual<TApi extends Api>(
	a: Model<TApi> | null | undefined,
	b: Model<TApi> | null | undefined,
): boolean {
	if (!a || !b) return false;
	return a.id === b.id && a.provider === b.provider;
}
