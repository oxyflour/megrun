// Browser and layout verification for the portable evidence slide deck.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL, fileURLToPath } = require('node:url');
const { chromium } = require('C:/Users/oxyfl/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'outputs', 'two_ideas_slides');
const qa = path.join(output, 'qa');
fs.mkdirSync(qa, { recursive: true });
const file = path.join(output, 'camera_magnetic_acceleration.html');
const browserErrors = [];
const checks = [];
function check(name, passed, detail) {
  checks.push({ name, passed: Boolean(passed), detail });
  if (!passed) throw new Error(name + ': ' + JSON.stringify(detail));
}

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
    const page = await context.newPage();
    page.on('pageerror', error => browserErrors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(pathToFileURL(file).href);
    await page.evaluate(() => document.fonts.ready);
    check('13 slides', await page.locator('.slide').count() === 13);
    check('all figures rendered', await page.locator('.chart svg').count() === 18,
          await page.locator('.chart svg').count());
    const slides = [];
    for (let index = 1; index <= 13; index++) {
      await page.evaluate(i => show(i - 1), index);
      await page.locator('.deck').screenshot({ path: path.join(qa, `slide-${String(index).padStart(2, '0')}.png`) });
      slides.push(await page.locator('.slide.active').evaluate(slide => {
        const bounds = slide.getBoundingClientRect();
        const outside = [];
        for (const el of slide.querySelectorAll('h1,h2,p,dd,dt,figcaption,.foot,td,th,.status,.metric,.metric-label,.metric-note,.statement,svg text')) {
          const rect = el.getBoundingClientRect();
          const parentSvg = el.closest('svg');
          const limits = parentSvg ? parentSvg.getBoundingClientRect() : bounds;
          if (rect.width && rect.height && (rect.left < limits.left - 2 || rect.right > limits.right + 2 || rect.top < limits.top - 2 || rect.bottom > limits.bottom + 2)) {
            outside.push({ text: el.textContent.trim().slice(0, 90), tag: el.tagName, container: parentSvg ? parentSvg.parentElement.id : 'slide' });
          }
        }
        const main = slide.querySelector('.main');
        const foot = slide.querySelector('.foot');
        const overlap = main && foot ? main.getBoundingClientRect().bottom > foot.getBoundingClientRect().top + 1 : false;
        return { title: slide.dataset.title, outside, mainFootOverlap: overlap };
      }));
    }
    check('no JavaScript errors', browserErrors.length === 0, browserErrors);
    check('precise total-force axis', await page.locator('#metricTotal svg').textContent().then(text => text.includes('33.18') && text.includes('33.22')));

    await page.keyboard.press('Home');
    check('Home navigation', await page.evaluate(() => window.__currentSlide) === 1);
    await page.keyboard.press('ArrowRight');
    check('arrow navigation', await page.evaluate(() => window.__currentSlide) === 2);
    await page.getByRole('button', { name: '俯视', exact: true }).click();
    check('CAD top view', await page.locator('[data-view=top]').getAttribute('aria-pressed') === 'true');
    await page.getByRole('button', { name: '三维视角', exact: true }).click();
    await page.keyboard.press('g');
    check('contents dialog', await page.locator('#tocOverlay').evaluate(el => el.classList.contains('open')));
    await page.locator('.toc-item').nth(8).click();
    check('contents navigation', await page.evaluate(() => window.__currentSlide) === 9);
    await page.keyboard.press('n');
    check('data provenance notes', await page.locator('#notesText').textContent().then(text => text.includes('SHA-256:')));
    await page.keyboard.press('Escape');
    check('dialog closes', await page.locator('.overlay.open').count() === 0);
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#dataBtn').click();
    const download = await downloadPromise;
    await download.saveAs(path.join(qa, download.suggestedFilename()));
    check('evidence download valid JSON', Boolean(JSON.parse(fs.readFileSync(path.join(qa, download.suggestedFilename()), 'utf8')).sources));
    await page.goto(pathToFileURL(file).href + '#slide-7');
    check('deep link', await page.evaluate(() => window.__currentSlide) === 7);

    const links = await page.locator('.source-list a').evaluateAll(nodes => nodes.map(el => ({ text: el.textContent, url: el.href })));
    check('original source links resolve', links.every(link => fs.existsSync(fileURLToPath(link.url))), links);
    await page.pdf({ path: path.join(qa, 'print-validation.pdf'), preferCSSPageSize: true, printBackground: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => show(8));
    await page.screenshot({ path: path.join(qa, 'mobile.png') });
    check('mobile deck within viewport', await page.locator('.deck').evaluate(el => {
      const r = el.getBoundingClientRect();
      return r.left >= -1 && r.right <= innerWidth + 1 && r.top >= 54 && r.bottom <= innerHeight + 1;
    }));
    const result = { generatedAt: new Date().toISOString(), file, checks, slides, browserErrors };
    fs.writeFileSync(path.join(qa, 'browser-verification.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ checks: checks.length, screenshots: slides.length, layoutFlags: slides.filter(s => s.outside.length || s.mainFootOverlap), qa }, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
