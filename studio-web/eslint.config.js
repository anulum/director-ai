// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — lint the host, tools and browser tests.

import js from "@eslint/js";
import parser from "@typescript-eslint/parser";
import hooks from "eslint-plugin-react-hooks";

export default [
    { ignores: ["build/**", "dist/**", "playwright-report/**", "test-results/**"] },
    js.configs.recommended,
    {
        files: ["**/*.{js,mjs}"],
        languageOptions: {
            globals: { console: "readonly", process: "readonly", URL: "readonly", fetch: "readonly" },
        },
    },
    {
        files: ["**/*.{ts,tsx}"],
        languageOptions: { parser },
        plugins: { "react-hooks": hooks },
        rules: {
            "no-undef": "off",
            "no-unused-vars": "off",
            "react-hooks/rules-of-hooks": "error",
            "react-hooks/exhaustive-deps": "error",
        },
    },
];
