import { describe, expect, it } from "vitest";
import {
	clampThinkingLevel,
	getAvailableThinkingLevels,
	getClaudeThinkingOffMode,
	getModel,
	getModels,
	getProviders,
	getThinkingCapability,
	inferThinkingCapability,
	supportsAdaptiveThinking,
	supportsMax,
	supportsThinkingOff,
	supportsXhigh,
} from "../src/models.js";
import {
	type Api,
	type Model,
	THINKING_ENCODING_KINDS,
	THINKING_OFF_MODES,
	THINKING_TIERS,
	type ThinkingLevel,
} from "../src/types.js";

const anthropicBase = getModel("anthropic", "claude-opus-4-6") as Model<Api>;

describe("getThinkingCapability", () => {
	it("returns the inferred capability for models without declared thinking", () => {
		const opus46 = getModel("anthropic", "claude-opus-4-6");
		expect(getThinkingCapability(opus46)).toEqual({
			levels: ["minimal", "low", "medium", "high", "max"],
			off: "disabled",
			encoding: { kind: "adaptive" },
		});
	});

	it("returns an off-only capability for non-reasoning models", () => {
		const capability = getThinkingCapability({ ...anthropicBase, reasoning: false });
		expect(capability.levels).toEqual([]);
		expect(getAvailableThinkingLevels({ ...anthropicBase, reasoning: false })).toEqual(["off"]);
	});

	it("returns a conservative capability for an undefined model", () => {
		const capability = getThinkingCapability(undefined);
		expect(capability.levels).toEqual([]);
		expect(capability.off).toBe("disabled");
	});

	describe("declared capability", () => {
		it("wins over inference for tiers, off mode and encoding", () => {
			const capability = getThinkingCapability({
				...anthropicBase,
				id: "deepseek-v4.1-flash",
				provider: "custom-gateway",
				thinking: {
					levels: ["minimal", "low", "medium", "high", "xhigh", "max"],
					off: "disabled",
					encoding: { kind: "effort", field: "output_config.effort" },
				},
			});

			expect(capability.levels).toEqual(["minimal", "low", "medium", "high", "xhigh", "max"]);
			expect(capability.encoding).toEqual({ kind: "effort", field: "output_config.effort" });
		});

		it("allows a model previously limited by inference to expose xhigh and max", () => {
			const model = {
				...anthropicBase,
				id: "deepseek-v4.1-flash",
				provider: "custom-gateway",
				thinking: {
					levels: ["minimal", "low", "medium", "high", "xhigh", "max"] as ThinkingLevel[],
					off: "disabled" as const,
					encoding: { kind: "effort" as const },
				},
			};

			expect(supportsMax(model)).toBe(true);
			expect(supportsXhigh(model)).toBe(true);
			expect(clampThinkingLevel(model, "max")).toBe("max");
		});

		it("merges a partial declaration over inference", () => {
			const capability = getThinkingCapability({
				...anthropicBase,
				thinking: { off: "between_tools" },
			});

			// Levels and encoding are inferred; only the off mode was declared.
			expect(capability.off).toBe("between_tools");
			expect(capability.levels).toEqual(["minimal", "low", "medium", "high", "max"]);
			expect(capability.encoding.kind).toBe("adaptive");
		});

		it("keeps declared encoding fields when only the kind is overridden", () => {
			const capability = getThinkingCapability({
				...anthropicBase,
				thinking: {
					levels: ["low", "high"],
					off: "disabled",
					encoding: { kind: "budget", field: "thinking.budget_tokens", budgets: { high: 8192 } },
				},
			});

			expect(capability.encoding).toEqual({
				kind: "budget",
				field: "thinking.budget_tokens",
				budgets: { high: 8192 },
			});
		});

		it("normalises unsorted, duplicated and unknown tiers", () => {
			const capability = getThinkingCapability({
				...anthropicBase,
				thinking: {
					// Unsorted, duplicated, and containing a value outside the known scale.
					levels: ["max", "low", "max", "high", "off", "ultra"] as unknown as ThinkingLevel[],
					off: "disabled",
					encoding: { kind: "effort" },
				},
			});

			expect(capability.levels).toEqual(["low", "high", "max"]);
		});

		it("falls back to the inferred off mode and encoding kind for unknown values", () => {
			const capability = getThinkingCapability({
				...anthropicBase,
				thinking: {
					levels: ["low"],
					off: "sometimes" as never,
					encoding: { kind: "telepathy" as never },
				},
			});

			expect(capability.off).toBe("disabled");
			expect(capability.encoding.kind).toBe("adaptive");
		});

		it("supports a declared capability that removes every tier", () => {
			const model = {
				...anthropicBase,
				thinking: { levels: [], off: "disabled" as const, encoding: { kind: "toggle" as const } },
			};

			expect(getAvailableThinkingLevels(model)).toEqual(["off"]);
			expect(clampThinkingLevel(model, "high")).toBe("off");
		});
	});
});

describe("capability helpers derive from getThinkingCapability", () => {
	it("keeps supportsMax consistent with the resolved tiers", () => {
		const model = { ...anthropicBase, id: "gpt-5.6-sol", provider: "openai" };
		expect(supportsMax(model)).toBe(getThinkingCapability(model).levels.includes("max"));
		expect(supportsXhigh(model)).toBe(getThinkingCapability(model).levels.includes("xhigh"));
	});

	it("keeps supportsThinkingOff consistent with the resolved off mode", () => {
		for (const provider of getProviders()) {
			for (const model of getModels(provider) as Model<Api>[]) {
				const capability = getThinkingCapability(model);
				expect(supportsThinkingOff(model), `${provider}/${model.id}`).toBe(capability.off !== "omit");
				expect(getClaudeThinkingOffMode(model), `${provider}/${model.id}`).toBe(capability.off);
			}
		}
	});

	it("keeps supportsAdaptiveThinking consistent with the resolved encoding", () => {
		for (const provider of getProviders()) {
			for (const model of getModels(provider) as Model<Api>[]) {
				const capability = getThinkingCapability(model);
				expect(supportsAdaptiveThinking(model), `${provider}/${model.id}`).toBe(
					capability.encoding.kind === "adaptive",
				);
			}
		}
	});

	it("derives available levels from the capability, in ascending order", () => {
		for (const provider of getProviders()) {
			for (const model of getModels(provider) as Model<Api>[]) {
				const capability = getThinkingCapability(model);
				const available = getAvailableThinkingLevels(model);
				const label = `${provider}/${model.id}`;

				if (!model.reasoning) {
					expect(available, label).toEqual(["off"]);
					continue;
				}

				const expected = capability.off === "omit" ? capability.levels : ["off", ...capability.levels];
				expect(available, label).toEqual(expected);
			}
		}
	});
});

describe("capability invariants across the registry", () => {
	const allModels: Array<Model<Api>> = getProviders().flatMap((provider) => getModels(provider) as Model<Api>[]);

	it("has at least one registered model", () => {
		expect(allModels.length).toBeGreaterThan(0);
	});

	it("only reports known, unique tiers in ascending order", () => {
		const rank = (tier: ThinkingLevel) => THINKING_TIERS.indexOf(tier);

		for (const model of allModels) {
			const { levels } = getThinkingCapability(model);
			const label = `${model.provider}/${model.id}`;

			expect(new Set(levels).size, label).toBe(levels.length);
			for (const level of levels) expect(THINKING_TIERS, label).toContain(level);
			const ranks = levels.map(rank);
			expect(
				[...ranks].sort((a, b) => a - b),
				label,
			).toEqual(ranks);
		}
	});

	it("only reports known off modes and encoding kinds", () => {
		for (const model of allModels) {
			const capability = getThinkingCapability(model);
			const label = `${model.provider}/${model.id}`;
			expect(THINKING_OFF_MODES, label).toContain(capability.off);
			expect(THINKING_ENCODING_KINDS, label).toContain(capability.encoding.kind);
		}
	});

	it("never clamps to a level the model does not advertise", () => {
		const requested: Array<ThinkingLevel | "off"> = ["off", ...THINKING_TIERS];

		for (const model of allModels) {
			const available = getAvailableThinkingLevels(model);
			const label = `${model.provider}/${model.id}`;

			for (const level of requested) {
				expect(available, `${label} @ ${level}`).toContain(clampThinkingLevel(model, level));
			}
		}
	});

	it("clamps an unsupported max to the highest advertised level", () => {
		const model = { ...anthropicBase, id: "claude-opus-4-5", provider: "anthropic" };
		expect(getAvailableThinkingLevels(model).at(-1)).toBe("high");
		expect(clampThinkingLevel(model, "max")).toBe("high");
		expect(clampThinkingLevel(model, "xhigh")).toBe("high");
	});

	it("clamps off to minimal when thinking cannot be disabled", () => {
		const opus55 = getModel("anthropic", "claude-opus-5-5");
		expect(supportsThinkingOff(opus55)).toBe(false);
		expect(clampThinkingLevel(opus55, "off")).toBe("minimal");
	});

	/**
	 * Golden snapshot of every registered model's capability. A new model family that
	 * only lands on the inference fallback shows up here as a reviewable diff.
	 */
	it("matches the recorded capability of every registered model", () => {
		const summary = Object.fromEntries(
			allModels
				.map((model): [string, string] => {
					const capability = getThinkingCapability(model);
					return [
						`${model.provider}/${model.id}`,
						`${capability.off} | ${capability.levels.join(",") || "-"} | ${capability.encoding.kind}`,
					];
				})
				.sort(([a], [b]) => a.localeCompare(b)),
		);

		expect(summary).toMatchSnapshot();
	});
});

describe("inferThinkingCapability", () => {
	it("matches the resolved capability when nothing is declared", () => {
		for (const provider of getProviders()) {
			for (const model of getModels(provider) as Model<Api>[]) {
				if (model.thinking) continue;
				expect(inferThinkingCapability(model), `${provider}/${model.id}`).toEqual(getThinkingCapability(model));
			}
		}
	});

	it("uses the API to choose a non-adaptive encoding", () => {
		const openaiModel = {
			...anthropicBase,
			id: "some-model",
			provider: "custom",
			api: "openai-completions",
		} as Model<Api>;
		const anthropicModel = {
			...anthropicBase,
			id: "some-model",
			provider: "custom",
			api: "anthropic-messages",
		} as Model<Api>;

		expect(inferThinkingCapability(openaiModel).encoding.kind).toBe("effort");
		expect(inferThinkingCapability(anthropicModel).encoding.kind).toBe("budget");
	});
});
