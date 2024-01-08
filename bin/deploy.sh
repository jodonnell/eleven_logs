git branch -D deploy;
git checkout -b deploy;
npm run build;
git add out.js;
git commit -m 'out.js';
git push origin deploy -f;
git checkout main;
