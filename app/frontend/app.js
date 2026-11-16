const API_BASE = window.location.origin; // same-origin: Flask serves this file too

/* ---------------------------------------------------------------------
   Global state
--------------------------------------------------------------------- */
let QUESTIONS = [];
const answers = {};          // { form_key: [selected option indices] }
let step = 0;                // current wizard step
let lastRecommendations = []; // full payload from /api/recommend
let recByOEid = {};          // destination_id -> recommendation object (has .score)
let userLocation = null;     // { country, currency } detected once, cached

const CURRENCIES = ["USD","EUR","GBP","INR","JPY","AUD","CAD","SGD","AED","CNY","CHF","ZAR","BRL","MXN"];

/* ---------------------------------------------------------------------
   Boot + routing
--------------------------------------------------------------------- */
async function boot(){
  try {
    const health = await fetch(`${API_BASE}/api/health`).then(r => r.json());
    setStatus(`connected — ${health.num_destinations} destinations loaded`, 'ok');
    animateStats();
  } catch (e) {
    setStatus('cannot reach API — is the Flask server running?', 'err');
  }
  try {
    QUESTIONS = await fetch(`${API_BASE}/api/questions`).then(r => r.json());
    QUESTIONS.forEach(q => answers[q.key] = []);
  } catch (e) {
    document.getElementById('wizard').innerHTML = '<div class="errbox">Could not load survey questions.</div>';
  }
  window.addEventListener('hashchange', route);
  document.getElementById('startBtn').onclick = () => {
    document.getElementById('surveyMain').scrollIntoView({behavior:'smooth'});
  };
  document.getElementById('retakeBtn').onclick = () => {
    QUESTIONS.forEach(q => answers[q.key] = []);
    lastRecommendations = [];
    recByOEid = {};
    document.getElementById('wizard').dataset.rendered = '';
    location.hash = '#/';
  };
  document.getElementById('backBtn').onclick = () => { location.hash = '#/results'; };
  detectLocation(); // fire-and-forget, used later on the destination page
  route();
}

function animateStats(){
  document.querySelectorAll('.stat-num').forEach(el => {
    const target = parseInt(el.dataset.target, 10);
    const dur = 1100;
    const t0 = performance.now();
    function tick(now){
      const p = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

function setStatus(text, cls){
  const el = document.getElementById('apiStatus');
  el.textContent = text;
  el.className = 'apistatus ' + cls;
}

function route(){
  const hash = location.hash || '#/';
  const views = {
    'view-home': hash === '#/' || hash === '',
    'view-results': hash === '#/results',
  };
  const destMatch = hash.match(/^#\/destination\/(\d+)$/);

  Object.entries(views).forEach(([id, show]) => {
    document.getElementById(id).hidden = !show;
  });
  document.getElementById('view-destination').hidden = !destMatch;

  if (hash === '#/' || hash === '') {
    if (QUESTIONS.length && document.getElementById('wizard').dataset.rendered !== '1') renderWizard();
  } else if (hash === '#/results') {
    if (!lastRecommendations.length) { location.hash = '#/'; return; }
    renderResults();
  } else if (destMatch) {
    renderDestination(parseInt(destMatch[1], 10));
  }
  window.scrollTo({top:0});
}

/* ---------------------------------------------------------------------
   Wizard (one question per screen, like a boarding-pass check-in flow)
--------------------------------------------------------------------- */
function renderWizard(){
  const wizard = document.getElementById('wizard');
  wizard.dataset.rendered = '1';
  wizard.innerHTML = `
    <div class="wizard-progress"><div class="wizard-progress-bar" id="wizProgBar"></div></div>
    <div class="wizard-body" id="wizBody"></div>
    <div class="wizard-nav">
      <button class="wizard-nav-btn" id="wizBack">← Back</button>
      <button class="wizard-nav-btn wizard-nav-next" id="wizNext">Next →</button>
    </div>`;
  document.getElementById('wizBack').onclick = () => { if (step>0){ step--; renderStep(); } };
  document.getElementById('wizNext').onclick = onWizardNext;
  step = 0;
  renderStep();
}

function renderStep(){
  const q = QUESTIONS[step];
  const body = document.getElementById('wizBody');
  const isGrid = q.options.length > 4;
  body.classList.remove('step-enter');
  void body.offsetWidth; // restart animation
  body.classList.add('step-enter');
  body.innerHTML = `
    <div class="wizard-step-label">QUESTION ${step+1} OF ${QUESTIONS.length}</div>
    <h2 class="wizard-question">${q.title}</h2>
    <div class="opts ${isGrid ? 'grid2':''}" id="wizOpts"></div>
    ${q.multi ? '<div class="hint">Select as many as apply.</div>' : ''}
  `;
  const opts = document.getElementById('wizOpts');
  q.options.forEach((text, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'opt' + (answers[q.key].includes(i) ? ' sel' : '');
    btn.textContent = text;
    btn.onclick = () => {
      if (q.multi) {
        const idx = answers[q.key].indexOf(i);
        if (idx > -1) { answers[q.key].splice(idx,1); btn.classList.remove('sel'); }
        else { answers[q.key].push(i); btn.classList.add('sel'); }
      } else {
        answers[q.key] = [i];
        [...opts.children].forEach(c => c.classList.remove('sel'));
        btn.classList.add('sel');
      }
      updateNextLabel();
    };
    opts.appendChild(btn);
  });

  document.getElementById('wizProgBar').style.width = `${((step+1)/QUESTIONS.length)*100}%`;
  document.getElementById('wizBack').disabled = step === 0;
  updateNextLabel();
}

function updateNextLabel(){
  const btn = document.getElementById('wizNext');
  const isLast = step === QUESTIONS.length - 1;
  btn.textContent = isLast ? 'See my matches →' : 'Next →';
}

async function onWizardNext(){
  const q = QUESTIONS[step];
  if (answers[q.key].length === 0) {
    // nudge instead of silently doing nothing
    document.getElementById('wizBody').querySelector('.wizard-question').style.color = 'var(--coral)';
    setTimeout(() => { document.getElementById('wizBody').querySelector('.wizard-question').style.color = ''; }, 400);
    return;
  }
  if (step < QUESTIONS.length - 1) {
    step++;
    renderStep();
  } else {
    await runModel();
  }
}

async function runModel(){
  const btn = document.getElementById('wizNext');
  btn.disabled = true;
  btn.textContent = 'Scoring 1,486 destinations…';
  try {
    const res = await fetch(`${API_BASE}/api/recommend`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({answers, top_k: 9})
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({error: 'Request failed'}));
      alert(err.error || 'Something went wrong scoring your answers.');
      return;
    }
    const data = await res.json();
    lastRecommendations = data.recommendations;
    recByOEid = {};
    lastRecommendations.forEach(r => recByOEid[r.destination_id] = r);
    location.hash = '#/results';
  } catch (e) {
    alert('Could not reach the API: ' + e.message);
  } finally {
    btn.disabled = false;
    updateNextLabel();
  }
}

/* ---------------------------------------------------------------------
   Results dashboard
--------------------------------------------------------------------- */
function renderResults(){
  const grid = document.getElementById('resultsGrid');
  grid.innerHTML = '';
  const maxScore = lastRecommendations[0].score;

  const heading = document.getElementById('resultsHeading');
  const regionIdx = (answers.region && answers.region[0]);
  const regionLabel = regionIdx === 0 ? 'in India' : regionIdx === 1 ? 'abroad' : 'for you';
  if (heading) heading.textContent = `Ranked ${regionLabel}`;

  lastRecommendations.forEach((r, i) => {
    const card = document.createElement('div');
    card.className = 'dcard';
    card.style.setProperty('--i', i);
    card.innerHTML = `
      <div class="dcard-media">
        <div class="dcard-rank">#${i+1} match</div>
        <img alt="${r.name}" loading="lazy">
      </div>
      <div class="dcard-body">
        <h3 class="dcard-name">${cityOnly(r.name)}</h3>
        <div class="dcard-country">${r.country || ''}</div>
        <div class="dcard-tags">
          ${r.top_interest ? tag(r.top_interest) : ''}
          ${r.budget_tier ? tag(r.budget_tier) : ''}
        </div>
        <div class="dcard-match">
          <div class="dcard-match-label"><span>Match confidence</span><span>${(r.score*100).toFixed(0)}%</span></div>
          <div class="dcard-bar-wrap"><div class="dcard-bar" style="width:${(r.score/maxScore*100).toFixed(0)}%"></div></div>
        </div>
      </div>`;
    card.onclick = () => { location.hash = `#/destination/${r.destination_id}`; };
    grid.appendChild(card);
    fetchWikiThumb(r.name).then(url => {
      if (url) card.querySelector('img').src = url;
      else card.querySelector('.dcard-media').style.background = placeholderGradient(r.destination_id);
    });
  });
}

function tag(text){ return `<span class="tag">${text}</span>`; }
function cityOnly(name){ return name.split(',')[0]; }
function placeholderGradient(seed){
  const hues = [(seed*47)%360, (seed*47+40)%360];
  return `linear-gradient(135deg, hsl(${hues[0]},35%,70%), hsl(${hues[1]},40%,55%))`;
}

/* ---------------------------------------------------------------------
   Destination detail page
--------------------------------------------------------------------- */
async function renderDestination(id){
  const el = document.getElementById('destContent');
  el.innerHTML = '<div class="empty">Loading destination…</div>';

  let dest;
  try {
    dest = await fetch(`${API_BASE}/api/destination/${id}`).then(r => r.json());
    if (dest.error) throw new Error(dest.error);
  } catch (e) {
    el.innerHTML = `<div class="errbox">Could not load this destination: ${e.message}</div>`;
    return;
  }

  const rec = recByOEid[id]; // may be undefined if page was opened directly

  el.innerHTML = `
    <div class="dest-gallery" id="destGallery">
      <img class="g-main" alt="${dest.name}">
      <img alt="">
      <img alt="">
    </div>
    <div class="dest-head">
      <div>
        <h1 class="dest-title">${cityOnly(dest.name)}</h1>
        <p class="dest-country">${dest.country || ''}</p>
      </div>
      ${rec ? `<div class="dest-match-pill">${(rec.score*100).toFixed(0)}% match</div>` : ''}
    </div>

    <div class="dest-columns">
      <div>
        <div class="section-label">What this place is known for</div>
        <div class="facts-grid">
          ${fact('Typically visited by', dest.companions)}
          ${fact('Popular season', dest.season)}
          ${fact('Climate', dest.climate)}
          ${fact('Usual budget style', dest.budget_tier)}
          ${fact('Common trip type', dest.trip_style)}
          ${fact('Top draw', dest.top_interest)}
        </div>
        ${rec && rec.why_matched && rec.why_matched.length ? whyMatchedBlock(rec) : ''}
      </div>
      <div id="budgetSlot"></div>
    </div>
  `;

  // gallery images
  fetchWikiGallery(dest.name).then(urls => {
    const imgs = document.querySelectorAll('#destGallery img');
    imgs.forEach((img, i) => {
      if (urls[i]) img.src = urls[i];
      else img.closest('.dest-gallery').style.background = placeholderGradient(id);
    });
  });

  // budget
  renderBudgetSlot(id);
}

function fact(label, value){
  return `<div class="fact"><div class="fact-label">${label.toUpperCase()}</div><div class="fact-value">${value || '—'}</div></div>`;
}

function whyMatchedBlock(rec){
  const hits = rec.why_matched.filter(w => w.matched).length;
  const rows = rec.why_matched.map(w => `
    <li class="why-row ${w.matched ? 'hit' : 'miss'}">
      <span class="why-mark">${w.matched ? '✓' : '✕'}</span>
      <span class="why-dim">${w.dimension}</span>
      <span class="why-val">${w.destination_value}</span>
    </li>`).join('');
  return `
    <div class="why-matched">
      <div class="section-label">Why this ranked where it did</div>
      <p class="why-sub">${hits} of ${rec.why_matched.length} things you picked line up with this place's own profile. This is the actual arithmetic behind the ${(rec.score*100).toFixed(0)}% — not a hidden model guess.</p>
      <ul class="why-list">${rows}</ul>
    </div>`;
}

async function renderBudgetSlot(destId){
  const slot = document.getElementById('budgetSlot');
  const hasAnswers = Object.values(answers).some(a => a.length > 0);
  if (!hasAnswers) {
    slot.innerHTML = `
      <div class="budget-card">
        <div class="section-label">Estimated trip budget</div>
        <p style="font-family:var(--sans);font-size:13.5px;color:#cfe3de;line-height:1.6;">
          Take the survey and this turns into a real number, built from your budget style,
          trip length and who you're travelling with.
        </p>
        <a href="#/" class="btn-ghost" style="border-color:#fff;color:#fff;">Take the survey</a>
      </div>`;
    return;
  }

  slot.innerHTML = `<div class="budget-card"><div class="section-label">Estimated trip budget</div><p style="font-family:var(--sans);font-size:13px;color:#cfe3de;">Calculating…</p></div>`;

  let est;
  try {
    est = await fetch(`${API_BASE}/api/estimate`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({answers, destination_id: destId})
    }).then(r => r.json());
    if (est.error) throw new Error(est.error);
  } catch (e) {
    slot.innerHTML = `<div class="budget-card"><div class="section-label">Estimated trip budget</div><div class="errbox">Could not estimate a budget: ${e.message}</div></div>`;
    return;
  }

  renderBudgetCard(slot, est, 'USD');
}

async function renderBudgetCard(slot, est, currency){
  const loc = userLocation;
  let converted = null;
  if (currency !== 'USD') {
    converted = await convertUSD(est.total_usd, currency);
  }
  const symbolTotal = converted != null ? fmtMoney(converted, currency) : fmtMoney(est.total_usd, 'USD');
  const rows = Object.entries(est.breakdown).map(([k,v]) => {
    return `<div class="budget-row"><span>${labelize(k)}</span><span>${fmtMoney(v, 'USD')}</span></div>`;
  }).join('');

  const currencyOptions = CURRENCIES.map(c => `<option value="${c}" ${c===currency?'selected':''}>${c}</option>`).join('');

  slot.innerHTML = `
    <div class="budget-card">
      <div class="section-label">Estimated trip budget</div>
      <div class="budget-total">${symbolTotal}</div>
      <div class="budget-sub">
        ${est.days} days · ${est.travellers} traveller${est.travellers>1?'s':''} · ${est.budget_style.toLowerCase()}
        ${loc ? `· detected near ${loc.country}` : ''}
      </div>
      <div class="budget-rows">${rows}</div>
      <div class="budget-note">
        Shown in USD per category above; total converted to your chosen currency.
        Scaled to this destination's own cost-of-travel level (×${est.cost_multiplier}
        for ${est.country || 'this destination'}), not just your budget style — a
        shoestring week here isn't priced the same as a shoestring week somewhere pricier.
        Rough estimate from stated preferences, not live prices or flights.
        <div style="margin-top:8px;">Currency:
          <select class="currency-select" id="currSelect">${currencyOptions}</select>
        </div>
      </div>
    </div>`;

  document.getElementById('currSelect').onchange = (e) => {
    renderBudgetCard(slot, est, e.target.value);
  };
}

function labelize(key){
  return {stay:'Accommodation', food:'Food & drink', activities:'Activities', local_transport:'Local transport'}[key] || key;
}
function fmtMoney(amount, currency){
  try {
    return new Intl.NumberFormat(undefined, {style:'currency', currency, maximumFractionDigits:0}).format(amount);
  } catch(e) {
    return `${currency} ${Math.round(amount).toLocaleString()}`;
  }
}

async function convertUSD(amountUsd, toCurrency){
  try {
    const r = await fetch(`https://api.frankfurter.app/latest?amount=${amountUsd}&from=USD&to=${toCurrency}`);
    const data = await r.json();
    return data.rates && data.rates[toCurrency] != null ? data.rates[toCurrency] : null;
  } catch (e) {
    return null;
  }
}

async function detectLocation(){
  try {
    const r = await fetch('https://ipapi.co/json/');
    const data = await r.json();
    if (data && data.country_name) {
      userLocation = { country: data.country_name, currency: data.currency || 'USD' };
      const ticket = document.getElementById('ticketFrom');
      if (ticket) ticket.textContent = userLocation.country;
    }
  } catch (e) {
    userLocation = null;
  }
}

/* ---------------------------------------------------------------------
   Wikipedia images — no API key required
--------------------------------------------------------------------- */
async function fetchWikiThumb(destName){
  const title = wikiTitleGuess(destName);
  try {
    const url = `https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageimages&piprop=thumbnail&pithumbsize=500&redirects=1&titles=${encodeURIComponent(title)}&origin=*`;
    const data = await fetch(url).then(r => r.json());
    const pages = data.query && data.query.pages;
    if (!pages) return null;
    const page = Object.values(pages)[0];
    return page && page.thumbnail ? page.thumbnail.source : null;
  } catch (e) {
    return null;
  }
}

async function fetchWikiGallery(destName){
  const title = wikiTitleGuess(destName);
  try {
    const url = `https://en.wikipedia.org/api/rest_v1/page/media-list/${encodeURIComponent(title)}`;
    const data = await fetch(url).then(r => { if (!r.ok) throw new Error('no media list'); return r.json(); });
    const items = (data.items || []).filter(it =>
      it.type === 'image' && it.srcset && it.srcset.length &&
      !/logo|flag|icon|map/i.test(it.title || '')
    );
    const urls = items.slice(0, 3).map(it => {
      const best = it.srcset[it.srcset.length - 1];
      let src = best.src;
      if (src.startsWith('//')) src = 'https:' + src;
      return src;
    });
    if (urls.length) return urls;
  } catch (e) { /* fall through to single-thumb fallback */ }

  const thumb = await fetchWikiThumb(destName);
  return thumb ? [thumb] : [];
}

function wikiTitleGuess(destName){
  // "Palermo, Sicily, Italy" -> try the city name first; Wikipedia handles most
  // city articles directly and redirects handle the rest.
  return destName.split(',')[0].trim();
}

boot();
