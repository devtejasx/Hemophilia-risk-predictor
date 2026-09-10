/**
 * ESLint configuration.
 *
 * package.json has declared a `lint` script since the frontend was written, but
 * no config file existed, so the command failed with "couldn't find a
 * configuration file" rather than linting anything. The TypeScript parser was
 * missing too: without it ESLint cannot read .ts/.tsx at all.
 *
 * The rule set is deliberately small. `tsc --noEmit` already covers types via
 * `npm run type-check`, so this focuses on what the compiler does not catch —
 * chiefly the react-hooks rules, which find stale-closure and missing-dependency
 * bugs in the data-fetching effects these pages are built from.
 */
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', 'node_modules', '.eslintrc.cjs', 'vite.config.ts'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
  },
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    // The API returns values whose shape is decided by the served model, not by
    // this client, so `any` is sometimes the honest type at the boundary.
    // Flagged rather than forbidden.
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unused-vars': [
      'error',
      { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
    ],
  },
}
