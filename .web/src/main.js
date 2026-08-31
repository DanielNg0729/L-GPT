const thread = document.getElementById('thread')
const form = document.getElementById('composer')
const input = document.getElementById('input')
const send = document.getElementById('send')
const layersEl = document.getElementById('layers')

let sessionId = null
let busy = false

const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))

const money = (p) =>
  p === null || p === undefined || p === '' ? null
    : (typeof p === 'number' ? `$${p.toFixed(2)}` : `${p}`.startsWith('$') ? p : `$${p}`)

/* Which layers the backend actually has live. Shown because "the models are enabled"
   and "the models loaded" are different claims, and only one of them is checkable. */
fetch('/api/health').then(r => r.json()).then(h => {
  const on = Object.entries(h.layers).filter(([, v]) => v).map(([k]) => k)
  // The payload contains the five configurable layers only. The exact catalogue span
  // node is deterministic core and is reported separately, not counted in this badge.
  const total = Object.keys(h.layers).length
  layersEl.innerHTML = on.length
    ? `${on.length}/${total} layers <b>live</b>`
    : `<b>deterministic</b> only`
  layersEl.title = Object.entries(h.layers)
    .map(([k, v]) => `${v ? '✓' : '·'} ${k}`).join('\n') + `\n\n${h.products.toLocaleString()} products`
}).catch(() => { layersEl.textContent = 'backend offline' })

/* A RAIL OF VERTICAL CARDS, WITH DETAIL BELOW IT.
   Ten full records stacked vertically buried the conversation -- one card was taller than
   the viewport. Ten summary ROWS fitted, but read as a spreadsheet. Vertical cards in a
   horizontal rail cost one card's height no matter how many there are, and the shape says
   "browse sideways" without a label.
   The full record opens in a panel UNDER the rail rather than inside a card, because
   growing one card in a horizontal strip shoves its neighbours off-screen. */
function summaryCard(p, rank) {
  const price = money(p.price)
  const cats = (p.categories || []).slice(-2).join(' › ')
  return `
    <button class="pcard" type="button" data-rank="${rank}" aria-pressed="false">
      <span class="rank">${rank}</span>
      <span class="pt">${esc(p.title)}</span>
      <span class="pm">
        ${price ? `<b class="price">${esc(price)}</b>` : ''}
        ${p.rating ? `<span>★ ${esc(p.rating)}</span>` : ''}
      </span>
      <span class="ps">${esc(p.store || '')}</span>
      <span class="pc">${esc(cats)}</span>
    </button>`
}

function detailPanel(p) {
  const det = Object.entries(p.details || {}).slice(0, 16)
  const feats = p.features || []
  const desc = (Array.isArray(p.description) ? p.description.join(' ') : p.description || '').trim()
  const price = money(p.price)
  return `
    <div class="detail-head">
      <h3>${esc(p.title)}</h3>
      <div class="sub">
        ${price ? `<span class="price">${esc(price)}</span>` : ''}
        ${p.store ? `<span>${esc(p.store)}</span>` : ''}
        ${p.rating ? `<span>★ ${esc(p.rating)} (${esc((p.rating_count ?? 0).toLocaleString())})</span>` : ''}
        <span>${esc(p.parent_asin)}</span>
      </div>
      ${(p.categories || []).length ? `<div class="crumb">${p.categories.map(c => esc(c)).join(' › ')}</div>` : ''}
    </div>
    <div class="facts">
      ${feats.length ? `<h4>Features</h4><ul>${feats.map(f => `<li>${esc(f)}</li>`).join('')}</ul>` : ''}
      ${det.length ? `<h4>Details</h4><dl>${det.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>` : ''}
      ${desc ? `<h4>Description</h4><p class="desc">${esc(desc)}</p>` : ''}
    </div>`
}

function render(data) {
  const [first, ...rest] = data.products || []
  const wrap = document.createElement('div')
  wrap.className = 'turn'

  /* THE CARD COMES FIRST, THEN THE SENTENCE. The agent's answer is the product; the
     message is the follow-up question about it. Putting the text first would read as
     though the question were the answer. */
  wrap.innerHTML = `
    <div class="copilot">
      ${first ? `<div class="rail">${summaryCard(first, 1)}<div class="rest" hidden>${
        rest.map((p, i) => summaryCard(p, i + 2)).join('')}</div></div>`
        : '<div class="say">I could not find a match for that yet.</div>'}
      ${rest.length ? `
        <button class="more" type="button" aria-expanded="false">
          <svg class="chev" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
            <path d="m9 6 6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.4"
                  stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          <span class="label">Show ${rest.length} more ${rest.length === 1 ? 'match' : 'matches'}</span>
        </button>` : ''}
      <div class="detail" hidden></div>
      <div class="say" style="margin-top:10px">${esc(data.message)}</div>
      <div class="meta">
        <span class="tag ${data.recognised ? '' : 'on'}">${data.recognised ? 'recognised wording' : 'unfamiliar wording → learned layers ran'}</span>
        ${data.ask_attribute ? `<span class="tag">asking about: ${esc(data.ask_attribute)}</span>` : ''}
        ${data.evidence?.length ? `<span class="tag">evidence: ${data.evidence.length}</span>` : ''}
        <span>turn ${data.turn}</span>
      </div>
    </div>`

  /* WHEEL OVER THE RAIL MOVES IT SIDEWAYS.
     The rail lives inside a vertically scrolling thread, so a wheel event has two
     plausible targets and the browser picks one -- which is why scrolling over the cards
     felt like it dropped input: some ticks moved the rail, some moved the page. Shift is
     the documented workaround and nobody reaches for it.
     So the rail claims vertical wheel deltas and turns them into horizontal movement, and
     RELEASES them at either end, so reaching the last card lets the page scroll on
     normally instead of trapping the pointer. `passive:false` because this preventDefaults. */
  const rail = wrap.querySelector('.rail')
  if (rail) {
    rail.addEventListener('wheel', (e) => {
      if (rail.scrollWidth <= rail.clientWidth) return          // nothing to scroll
      // The rail owns the wheel outright while the pointer is over it: any direction, any
      // modifier, moves it sideways and nothing else. Releasing at the ends was tried and
      // was worse -- reaching the last card handed the page a scroll mid-gesture, so one
      // motion did two different things depending on where the rail happened to be.
      e.preventDefault()
      const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY
      // deltaMode 1 is lines, 2 is pages; normalise so a trackpad and a wheel agree.
      const step = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? rail.clientWidth : 1
      rail.scrollLeft += d * step
    }, { passive: false })
  }

  /* Delegated, so cards revealed by "show more" get the behaviour without rebinding. */
  const detail = wrap.querySelector('.detail')
  wrap.addEventListener('click', (e) => {
    const card = e.target.closest('.pcard')
    if (!card || !wrap.contains(card)) return
    const rank = Number(card.dataset.rank)
    const already = card.getAttribute('aria-pressed') === 'true'
    wrap.querySelectorAll('.pcard').forEach((c) => c.setAttribute('aria-pressed', 'false'))
    if (already) {                       // clicking the open card closes it
      detail.hidden = true
      detail.innerHTML = ''
      return
    }
    card.setAttribute('aria-pressed', 'true')
    detail.innerHTML = detailPanel(data.products[rank - 1])
    detail.hidden = false
  })

  const btn = wrap.querySelector('.more')
  if (btn) {
    btn.addEventListener('click', () => {
      const panel = wrap.querySelector('.rest')
      const open = btn.getAttribute('aria-expanded') === 'true'
      btn.setAttribute('aria-expanded', String(!open))
      panel.hidden = open
      btn.querySelector('.label').textContent =
        open ? `Show ${rest.length} more ${rest.length === 1 ? 'match' : 'matches'}` : 'Hide other matches'
    })
  }
  thread.appendChild(wrap)
  thread.scrollTop = thread.scrollHeight
}

function saySelf(text) {
  const d = document.createElement('div')
  d.className = 'turn'
  d.innerHTML = `<div class="shopper">${esc(text)}</div>`
  thread.appendChild(d)
  thread.scrollTop = thread.scrollHeight
}

async function ask(text) {
  if (busy || !text.trim()) return
  busy = true
  send.disabled = true
  document.querySelector('.intro')?.remove()
  saySelf(text)

  const pending = document.createElement('div')
  pending.className = 'turn'
  pending.innerHTML = `<div class="copilot"><div class="say"><span class="dots"><i></i><i></i><i></i></span></div></div>`
  thread.appendChild(pending)
  thread.scrollTop = thread.scrollHeight

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    })
    const data = await res.json()
    sessionId = data.session_id
    pending.remove()
    render(data)
  } catch (err) {
    pending.querySelector('.say').textContent =
      'The agent is not reachable. Start it with: python .web/server.py'
  } finally {
    busy = false
    send.disabled = false
    input.focus()
  }
}

function submit() {
  const text = input.value
  input.value = ''
  ask(text)
}

form.addEventListener('submit', (e) => {
  e.preventDefault()
  submit()
})

/* Enter sends. Implicit form submission is supposed to cover this, and in testing it did
   not fire, so the chat's most-used key is handled explicitly rather than left to depend
   on it. Shift+Enter is left alone for anyone who later swaps the input for a textarea. */
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    submit()
  }
})

document.querySelectorAll('.chips button').forEach((b) =>
  b.addEventListener('click', () => ask(b.textContent.trim())))

document.getElementById('reset').addEventListener('click', async () => {
  if (sessionId) {
    await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: '' }),
    }).catch(() => {})
  }
  sessionId = null
  location.reload()
})

input.focus()
