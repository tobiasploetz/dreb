import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getThinkingCapability } from "@dreb/ai";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.js";
import { clearApiKeyCache, ModelRegistry } from "../src/core/model-registry.js";

describe("models.json thinking capability", () => {
	let tempDir: string;
	let modelsJsonPath: string;
	let authStorage: AuthStorage;

	beforeEach(() => {
		tempDir = join(tmpdir(), `dreb-test-thinking-capability-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
		modelsJsonPath = join(tempDir, "models.json");
		authStorage = AuthStorage.create(join(tempDir, "auth.json"));
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
		clearApiKeyCache();
	});

	function writeModelsJson(providers: Record<string, unknown>) {
		writeFileSync(modelsJsonPath, JSON.stringify({ providers }));
	}

	function loadRegistry(): ModelRegistry {
		const registry = new ModelRegistry(authStorage, modelsJsonPath);
		return registry;
	}

	test("per-model override declares tiers the inference would not allow", () => {
		writeModelsJson({
			anthropic: {
				modelOverrides: {
					"claude-opus-4-5": {
						thinking: {
							levels: ["minimal", "low", "medium", "high", "xhigh", "max"],
							off: "disabled",
							encoding: { kind: "effort", field: "output_config.effort" },
						},
					},
				},
			},
		});

		const registry = loadRegistry();
		expect(registry.getError()).toBeUndefined();

		const model = registry.find("anthropic", "claude-opus-4-5");
		expect(model).toBeDefined();

		const capability = getThinkingCapability(model!);
		expect(capability.levels).toEqual(["minimal", "low", "medium", "high", "xhigh", "max"]);
		expect(capability.encoding).toEqual({ kind: "effort", field: "output_config.effort" });
	});

	test("per-model override merges over the inferred capability", () => {
		writeModelsJson({
			anthropic: {
				modelOverrides: {
					"claude-opus-4-6": { thinking: { off: "between_tools" } },
				},
			},
		});

		const registry = loadRegistry();
		const capability = getThinkingCapability(registry.find("anthropic", "claude-opus-4-6")!);

		expect(capability.off).toBe("between_tools");
		// Inference still supplies the tiers and the adaptive encoding.
		expect(capability.levels).toEqual(["minimal", "low", "medium", "high", "max"]);
		expect(capability.encoding.kind).toBe("adaptive");
	});

	test("custom model definition carries its declared capability", () => {
		writeModelsJson({
			"local-gateway": {
				baseUrl: "http://127.0.0.1:9999",
				apiKey: "LOCAL_KEY",
				api: "anthropic-messages",
				models: [
					{
						id: "deepseek-v4.1-flash",
						name: "DeepSeek V4.1 Flash",
						reasoning: true,
						thinking: {
							levels: ["minimal", "low", "medium", "high", "xhigh", "max"],
							off: "disabled",
							encoding: { kind: "effort" },
						},
					},
				],
			},
		});

		const registry = loadRegistry();
		expect(registry.getError()).toBeUndefined();

		const model = registry.find("local-gateway", "deepseek-v4.1-flash");
		expect(getThinkingCapability(model!).levels).toContain("max");
	});

	test("rejects thinking on a non-reasoning model", () => {
		writeModelsJson({
			anthropic: {
				modelOverrides: {
					"claude-opus-4-6": { reasoning: false, thinking: { levels: ["high"] } },
				},
			},
		});

		const registry = loadRegistry();
		expect(registry.getError()).toContain('"thinking" cannot be declared for a non-reasoning model');
	});

	test("rejects duplicate levels", () => {
		writeModelsJson({
			anthropic: {
				modelOverrides: {
					"claude-opus-4-6": { thinking: { levels: ["high", "high"] } },
				},
			},
		});

		const registry = loadRegistry();
		expect(registry.getError()).toContain("must not contain duplicates");
	});

	test("rejects an unknown tier through schema validation", () => {
		writeModelsJson({
			anthropic: {
				modelOverrides: {
					"claude-opus-4-6": { thinking: { levels: ["turbo"] } },
				},
			},
		});

		const registry = loadRegistry();
		expect(registry.getError()).toContain("Invalid models.json schema");
	});

	test("rejects an unknown encoding kind through schema validation", () => {
		writeModelsJson({
			anthropic: {
				modelOverrides: {
					"claude-opus-4-6": { thinking: { encoding: { kind: "telepathy" } } },
				},
			},
		});

		const registry = loadRegistry();
		expect(registry.getError()).toContain("Invalid models.json schema");
	});
});
