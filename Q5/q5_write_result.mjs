import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

const [templatePath, bestPath, contributionPath, outputPath, previewPath] = process.argv.slice(2);
if (!previewPath) {
  throw new Error("用法：node q5_write_result.mjs 模板.xlsx 最优解.csv 边际贡献.csv 输出.xlsx 预览.png");
}

const [bestText, contributionText] = await Promise.all([
  fs.readFile(bestPath, "utf8"),
  fs.readFile(contributionPath, "utf8"),
]);
const best = parseCsv(bestText);
const contributions = parseCsv(contributionText);
const contributionMap = new Map(contributions.map((row) => [`${row.uav}-${row.bomb}`, row]));
const bestMap = new Map(best.map((row) => [`${row.uav}-${row.bomb}`, row]));

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const sheet = workbook.worksheets.getItem("Sheet1");
for (let rowNumber = 2; rowNumber <= 16; rowNumber += 1) {
  const uav = String(sheet.getRange(`A${rowNumber}`).values[0][0]);
  const bomb = String(sheet.getRange(`D${rowNumber}`).values[0][0]);
  const key = `${uav}-${bomb}`;
  const result = bestMap.get(key);
  const contribution = contributionMap.get(key);
  const used = contribution?.used === "True";
  if (!used || !result) {
    sheet.getRange(`B${rowNumber}:C${rowNumber}`).clear({ applyTo: "contents" });
    sheet.getRange(`E${rowNumber}:L${rowNumber}`).clear({ applyTo: "contents" });
    continue;
  }
  const missile = contribution.primary_missile;
  const individualKey = `${missile}_individual_s`;
  sheet.getRange(`B${rowNumber}:C${rowNumber}`).values = [[
    Number(result.heading_deg),
    Number(result.speed_mps),
  ]];
  sheet.getRange(`E${rowNumber}:L${rowNumber}`).values = [[
    Number(result.release_x_m),
    Number(result.release_y_m),
    Number(result.release_z_m),
    Number(result.burst_x_m),
    Number(result.burst_y_m),
    Number(result.burst_z_m),
    Number(result[individualKey]),
    missile,
  ]];
}
sheet.getRange("B2:K16").format.numberFormat = "0.000000";

const inspected = await workbook.inspect({
  kind: "table",
  sheetId: "Sheet1",
  range: "A1:L18",
  tableMaxRows: 18,
  tableMaxCols: 12,
  maxChars: 16000,
});
process.stdout.write(`${inspected.ndjson}\n`);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "result3 formula error scan",
});
process.stdout.write(`${errors.ndjson}\n`);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
const preview = await workbook.render({
  sheetName: "Sheet1",
  range: "A1:L18",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
process.stdout.write(`saved:${outputPath}\npreview:${previewPath}\n`);
