git checkout deploy;
npm run build;
git add out.js;
git commit -m 'out.js';
git push origin deploy;
git checkout main;
