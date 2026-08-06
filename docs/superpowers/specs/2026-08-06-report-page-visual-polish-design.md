# Report Page Visual Polish Design

## Goal

Improve the completed evaluation report page so operators can scan results quickly, inspect the smoothness visualization, review structured metadata, and download every existing report artifact. Preserve all current report calculations, filters, security checks, and download behavior.

## Scope

- Modify only the report presentation and the view data needed to describe downloadable files.
- Keep the existing report URL, filter query parameters, artifact whitelist, and download URLs.
- Render the existing smoothness_curve.svg inline when the artifact is available.
- Keep smoothness_curve.svg available as a normal attachment download.
- Do not change metric generation, evaluation execution, database models, or artifact formats.

## Page Structure

The report will use the approved layered operations layout.

1. **Header and core metrics**
   - Keep the dataset name and report context.
   - Present GSR, success/failure counts, successful TTS, and pending review as the primary summary.
   - Present smoothness as supporting context next to the chart rather than as a crowded summary-grid item.

2. **Smoothness and provenance**
   - Use a two-column desktop band.
   - The larger column previews smoothness_curve.svg.
   - The smaller column renders provenance as a semantic key/value table.
   - On narrow screens, stack the chart above the provenance table.
   - If the SVG is unavailable, omit the preview without breaking the rest of the report.

3. **Result files**
   - Show the output path with the existing copy action.
   - Replace the row of large filename buttons with a semantic table containing filename, description, format, and a download action.
   - Every artifact currently returned by _available_downloads remains downloadable.
   - Use concise, human-readable descriptions based on the whitelisted filename.

4. **Episode details**
   - Retain the existing result and review filters.
   - Retain all current columns and VLM behavior.
   - Improve spacing, alignment, status presentation, and responsive overflow without changing filtering semantics.

5. **Related navigation**
   - Keep links back to the evaluation task and dataset.

## SVG Preview Safety

The browser will load the smoothness image through the existing authenticated artifact route, not from a filesystem path. The route already restricts filenames, resolves containment, and serves only whitelisted artifacts. The template will use the download URL as an image source and will not inject SVG markup into the page.

## Download Metadata

_available_downloads will continue returning the artifact name and URL, with two display-only fields added:

- description: a fixed label for the artifact's purpose.
- format: a short uppercase extension label.

Unknown future whitelisted report markdown filenames will receive the report description and MD format. No user-controlled HTML is introduced.

## Styling

- Extend the existing restrained green, neutral, work-focused design.
- Use full-width bands and tables; do not introduce nested cards.
- Keep border radii at 6px or less.
- Use stable grid constraints and horizontal table scrolling on narrow screens.
- Use the existing icon library for copy and download actions.
- Keep all visible text readable without overlap at desktop and mobile breakpoints.

## Error Handling

- Missing optional SVG: show no preview and keep all other report sections operational.
- Missing required metrics_core.json: retain the existing 404 behavior.
- Malformed optional episode CSV: retain the current graceful degradation.
- Missing or unsafe download: retain the existing 404 behavior.

## Testing

Use test-driven development with focused coverage:

- The report page renders the SVG preview only when the artifact exists.
- The result-file table includes all existing download URLs and display metadata.
- Existing downloads still return attachments with unchanged content.
- Existing report filters and VLM episode rendering continue to pass.
- Responsive layout receives the existing visual-layout checks plus a targeted report-page browser assertion if supported by the current E2E harness.

Run the focused report tests, the full test suite, Ruff, and git diff --check. Restart the local Web server and verify the real report page and each download link.
