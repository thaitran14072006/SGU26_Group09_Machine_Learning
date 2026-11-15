// Export the CSV experiment log to Excel. Requires @oai/artifact-tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
let api;
try { api = await import('@oai/artifact-tool'); }
catch {
  const modules = process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES || path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
  api = await import(pathToFileURL(path.join(modules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
}
const { Workbook, SpreadsheetFile } = api;
const dir = path.join(root, 'reports/individual');
const csv = (await fs.readFile(path.join(dir, 'nhat_ky_thuc_nghiem.csv'), 'utf8')).replace(/^\uFEFF/, '');
const wb = await Workbook.fromCSV(csv, {sheetName: 'Nhat ky Thai'});
const sheet = wb.worksheets.getItemAt(0);
const values = sheet.getUsedRange().values;
for (let r = 1; r < values.length; r++) {
  const [day, month, year] = values[r][0].split('/').map(Number);
  // Excel serial date, preserving the user's assigned experiment date.
  values[r][0] = (Date.UTC(year, month - 1, day) - Date.UTC(1899, 11, 30)) / 86400000;
  for (let c = 7; c <= 10; c++) values[r][c] = Number(values[r][c]);
}
sheet.getUsedRange().values = values;
const end = values.length;
sheet.showGridLines = false;
sheet.getRange(`A1:Q${end}`).format.font = {name:'Arial', size:11};
sheet.getRange(`A1:Q${end}`).format.verticalAlignment = 'center';
sheet.getRange(`A1:Q${end}`).format.columnWidth = 20;
sheet.getRange(`A2:Q${end}`).format.rowHeight = 76;
sheet.getRange(`A2:Q${end}`).format.wrapText = true;
sheet.getRange('A1:Q1').format = {fill:'#243B53', font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:40,horizontalAlignment:'center'};
sheet.getRange(`A2:A${end}`).setNumberFormat('dd/mm/yyyy');
sheet.getRange(`H2:I${end}`).setNumberFormat('0.00000');
sheet.getRange(`J2:J${end}`).setNumberFormat('#,##0');
sheet.getRange(`K2:K${end}`).setNumberFormat('0.00');
for (const col of ['F','G','L']) sheet.getRange(`${col}1:${col}${end}`).format.columnWidth = 42;
sheet.getRange(`M1:N${end}`).format.columnWidth = 34;
sheet.getRange(`O1:Q${end}`).format.columnWidth = 30;
for (let r=1; r<end; r++) {
  if (r%2===0) sheet.getRange(`A${r+1}:Q${r+1}`).format.fill = '#F1F5F9';
  for (const c of [12,13]) {
    const label = c===12 ? path.basename(values[r][c]) : `${values[r][1]} / result.json`;
    const target = '../../'+values[r][c];
    sheet.getCell(r,c).formulas = [[`=HYPERLINK("${target}","${label}")`]];
    sheet.getCell(r,c).format.font.color = '#1565C0';
  }
}
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);
wb.recalculate();
const check = await wb.inspect({kind:'table',range:'Nhat ky Thai!A1:D3',include:'values',tableMaxRows:3,tableMaxCols:4});
console.log(check.ndjson);
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:10},summary:'Formula check'});
console.log(errors.ndjson);
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(dir,'nhat_ky_thuc_nghiem.xlsx'));
if (process.env.THAI_LOG_PREVIEW) {
  const preview = await wb.render({sheetName:sheet.name,range:'A1:E6',scale:1.5,format:'png'});
  await fs.writeFile(process.env.THAI_LOG_PREVIEW,new Uint8Array(await preview.arrayBuffer()));
}
console.log(`Saved ${end-1} experiment records.`);
process.exit(0);
