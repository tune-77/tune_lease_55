import { ESLint } from "eslint";

const BASELINE = Object.freeze({
  totalWarnings: 304,
  byRule: {
    "@typescript-eslint/no-explicit-any": 205,
    "@typescript-eslint/no-unused-vars": 59,
    "@next/next/no-img-element": 17,
    "react-hooks/set-state-in-effect": 12,
    "react-hooks/exhaustive-deps": 10,
    "<eslint>": 1,
  },
});

const eslint = new ESLint();
const results = await eslint.lintFiles(["src/"]);
const actual = { totalWarnings: 0, totalErrors: 0, byRule: {} };

for (const result of results) {
  actual.totalWarnings += result.warningCount;
  actual.totalErrors += result.errorCount;
  for (const message of result.messages) {
    const rule = message.ruleId || "<eslint>";
    actual.byRule[rule] = (actual.byRule[rule] || 0) + 1;
  }
}

const regressions = [];
if (actual.totalErrors > 0) {
  regressions.push(`errors: ${actual.totalErrors} (expected 0)`);
}
if (actual.totalWarnings > BASELINE.totalWarnings) {
  regressions.push(`warnings: ${actual.totalWarnings} (baseline ${BASELINE.totalWarnings})`);
}
for (const [rule, count] of Object.entries(actual.byRule)) {
  const allowed = BASELINE.byRule[rule] || 0;
  if (count > allowed) {
    regressions.push(`${rule}: ${count} (baseline ${allowed})`);
  }
}

console.log(`ESLint debt: ${actual.totalWarnings}/${BASELINE.totalWarnings} warnings, ${actual.totalErrors} errors`);
for (const [rule, count] of Object.entries(actual.byRule).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${rule}: ${count}/${BASELINE.byRule[rule] || 0}`);
}

if (regressions.length > 0) {
  console.error("\nLint baseline regression detected:");
  for (const regression of regressions) console.error(`- ${regression}`);
  process.exitCode = 1;
}
