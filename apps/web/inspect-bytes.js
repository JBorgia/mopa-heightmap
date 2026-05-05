const fs = require('fs');
const c = fs.readFileSync('src/app/features/wizard/wizard-shell.component.ts', 'utf8');

const nonAscii = c.match(/[^\x00-\x7F]+/g) || [];
const unique = [...new Set(nonAscii)];
let out = '';
unique.forEach(seq => {
  const cp = Array.from(seq).map(ch => 'U+' + ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')).join(' ');
  out += JSON.stringify(seq) + ' -> ' + cp + '\n';
});
fs.writeFileSync('inspect-out.txt', out, 'utf8');
console.log('Written to inspect-out.txt');
