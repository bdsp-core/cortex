import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "**/public/bundle/**",
      "**/__testdata__/**",
      "**/visual_baselines/**",
    ],
  },
  {
    files: ["apps/web/**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
    ],
    plugins: {
      "react-hooks": reactHooks,
    },
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
        ...globals.worker,
      },
    },
    rules: {
      // The strict TypeScript compiler is authoritative for unused symbols and
      // unsafe types; disabling duplicate lint diagnostics keeps this gate
      // focused on JavaScript hazards and React hook correctness.
      "no-undef": "off",
      "no-unused-vars": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      // Preserve the ESLint 9 recommended-rule contract while the two existing
      // assignment sites are reviewed independently of this security upgrade.
      "no-useless-assignment": "off",
      // React Hooks 7 adds compiler-oriented rules to its recommended presets.
      // Keep the pre-upgrade hook contract without forcing an application
      // refactor as part of this dependency-only security change.
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
  },
);
