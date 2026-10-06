/**
 * app.js - Full-Stack Client Application for QuickBasket AI
 * Manages State, JWT Authentication, Catalog Filtering, Live Cart Drawer,
 * AI Recipe Copilot, 2D Semantic Canvas, ABSA Sentiment, and 10-Min Fast-Track Checkout.
 */

// ==========================================
// APPLICATION STATE
// ==========================================
const state = {
  catalog: [],
  aisles: [],
  cart: {}, // { [productId]: quantity }
  token: localStorage.getItem('quickbasket_token') || null,
  user: null,
  currentAisle: 'all',
  searchQuery: '',
  sortMode: 'default',
  activeTab: 'store',
  semanticData: null,
  canvasTransform: { scale: 1, offsetX: 0, offsetY: 0 },
  hoveredNode: null,
  activeProductModalId: null,
  currentCopilotKit: null,
  appliedPromo: null
};

// ==========================================
// UTILITY HELPERS & TOASTS
// ==========================================
function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '⚠️' : 'ℹ️'}</span> <span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

async function apiFetch(endpoint, options = {}) {
  const headers = options.headers || {};
  if (!headers['Content-Type'] && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  if (state.token) {
    headers['Authorization'] = `Bearer ${state.token}`;
  }
  options.headers = headers;

  try {
    const res = await fetch(endpoint, options);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(errData.detail || `Server responded with ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Error on ${endpoint}:`, err);
    throw err;
  }
}

// ==========================================
// INITIALIZATION
// ==========================================
document.addEventListener('DOMContentLoaded', async () => {
  console.log("⚡ [QuickBasket AI] Initializing client...");
  await initAuth();
  await loadCatalogAndAisles();
  await loadCartFromStorageOrDb();
  setupGlobalSearch();
  initCanvasListeners();
});

// ==========================================
// AUTHENTICATION MODULE
// ==========================================
async function initAuth() {
  if (state.token) {
    try {
      const user = await apiFetch('/api/auth/me');
      state.user = user;
      updateAuthUI();
    } catch (e) {
      console.warn("Session expired or invalid token. Clearing auth.");
      logoutUser(false);
    }
  } else {
    updateAuthUI();
  }
}

function updateAuthUI() {
  const authSection = document.getElementById('auth-section');
  if (!authSection) return;

  if (state.user) {
    authSection.innerHTML = `
      <div class="user-pill" onclick="switchTab('account')">
        <img class="user-pill-avatar" src="${state.user.avatar_url || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150'}" alt="User">
        <span class="user-pill-name">${state.user.full_name.split(' ')[0]}</span>
        <span class="user-pill-points">🪙 ${state.user.loyalty_points}</span>
      </div>
    `;
  } else {
    authSection.innerHTML = `
      <button id="btn-login-open" class="btn btn-secondary btn-sm" onclick="openAuthModal('login')">
        <span>👤 Sign In / Join</span>
      </button>
    `;
  }

  // Update account tab view
  updateAccountView();
}

function openAuthModal(mode = 'login') {
  setAuthMode(mode);
  const modal = document.getElementById('auth-modal');
  if (modal) modal.style.display = 'flex';
}

function closeAuthModal() {
  const modal = document.getElementById('auth-modal');
  if (modal) modal.style.display = 'none';
  const errBox = document.getElementById('auth-error-alert');
  if (errBox) errBox.style.display = 'none';
}

let currentAuthMode = 'login';
function setAuthMode(mode) {
  currentAuthMode = mode;
  const tabLogin = document.getElementById('tab-login-btn');
  const tabReg = document.getElementById('tab-register-btn');
  const title = document.getElementById('auth-modal-title');
  const submitBtn = document.getElementById('auth-submit-btn');
  const fullnameGroup = document.getElementById('auth-fullname-group');
  const idLabel = document.getElementById('auth-identifier-label');
  const idInput = document.getElementById('auth-identifier');

  if (mode === 'login') {
    tabLogin.classList.add('active');
    tabReg.classList.remove('active');
    title.innerText = 'Sign In to QuickBasket AI';
    submitBtn.innerText = 'Sign In';
    fullnameGroup.style.display = 'none';
    idLabel.innerText = 'Email or Username:';
    idInput.placeholder = 'demo@antigravity.ai';
  } else {
    tabLogin.classList.remove('active');
    tabReg.classList.add('active');
    title.innerText = 'Create Shopper Account';
    submitBtn.innerText = 'Register Account';
    fullnameGroup.style.display = 'flex';
    idLabel.innerText = 'Email Address:';
    idInput.placeholder = 'you@example.com';
  }
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  const errBox = document.getElementById('auth-error-alert');
  errBox.style.display = 'none';

  const idVal = document.getElementById('auth-identifier').value.trim();
  const pwdVal = document.getElementById('auth-password').value;

  try {
    let result;
    if (currentAuthMode === 'login') {
      result = await apiFetch('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username_or_email: idVal, password: pwdVal })
      });
    } else {
      const nameVal = document.getElementById('auth-fullname').value.trim() || 'Quick Shopper';
      const username = idVal.split('@')[0] + Math.floor(Math.random() * 900 + 100);
      result = await apiFetch('/api/auth/register', {
        method: 'POST',
        body: JSON.stringify({ email: idVal, username, password: pwdVal, full_name: nameVal })
      });
    }

    state.token = result.access_token;
    state.user = result.user;
    localStorage.setItem('quickbasket_token', state.token);
    closeAuthModal();
    updateAuthUI();
    showToast(`Welcome back, ${state.user.full_name}!`, 'success');
    await syncCartWithBackend();
  } catch (err) {
    errBox.innerText = err.message || 'Authentication failed. Please verify credentials.';
    errBox.style.display = 'block';
  }
}

async function quickDemoLogin() {
  try {
    const result = await apiFetch('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username_or_email: 'demo@antigravity.ai', password: 'Password123!' })
    });
    state.token = result.access_token;
    state.user = result.user;
    localStorage.setItem('quickbasket_token', state.token);
    closeAuthModal();
    updateAuthUI();
    showToast('Signed in as Chef Alex Rivera (Demo Shopper)', 'success');
    await syncCartWithBackend();
  } catch (err) {
    showToast('Demo login error: ' + err.message, 'error');
  }
}

function logoutUser(toast = true) {
  state.token = null;
  state.user = null;
  localStorage.removeItem('quickbasket_token');
  updateAuthUI();
  if (toast) showToast('Logged out successfully.', 'info');
}

// ==========================================
// CATALOG & AISLE MODULE
// ==========================================
async function loadCatalogAndAisles() {
  try {
    const [products, aisles] = await Promise.all([
      apiFetch('/api/catalog/products'),
      apiFetch('/api/catalog/aisles')
    ]);
    state.catalog = products;
    state.aisles = aisles;

    renderAislePills();
    renderProductGrid();
    populateABSAPicker();
  } catch (e) {
    showToast('Failed to load catalog inventory: ' + e.message, 'error');
  }
}

function renderAislePills() {
  const container = document.getElementById('aisle-pills-container');
  if (!container) return;

  let html = `
    <button class="aisle-pill ${state.currentAisle === 'all' ? 'active' : ''}" onclick="filterAisle('all')">
      <span class="aisle-emoji">🏷️</span> All Aisles (${state.catalog.length})
    </button>
  `;

  for (const a of state.aisles) {
    const isActive = state.currentAisle.toLowerCase() === a.name.toLowerCase();
    html += `
      <button class="aisle-pill ${isActive ? 'active' : ''}" onclick="filterAisle('${a.name.replace(/'/g, "\\'")}')">
        <span class="aisle-emoji">${a.icon}</span> ${a.name} (${a.count})
      </button>
    `;
  }
  container.innerHTML = html;
}

function filterAisle(aisleName) {
  state.currentAisle = aisleName;
  renderAislePills();
  renderProductGrid();
}

function setupGlobalSearch() {
  const input = document.getElementById('global-search');
  const clearBtn = document.getElementById('search-clear-btn');
  if (!input) return;

  let debounceTimer;
  input.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      state.searchQuery = e.target.value.trim().toLowerCase();
      if (clearBtn) clearBtn.style.display = state.searchQuery ? 'block' : 'none';
      renderProductGrid();
    }, 200);
  });
}

function clearSearch() {
  const input = document.getElementById('global-search');
  const clearBtn = document.getElementById('search-clear-btn');
  if (input) input.value = '';
  if (clearBtn) clearBtn.style.display = 'none';
  state.searchQuery = '';
  renderProductGrid();
}

function sortCatalog(mode) {
  state.sortMode = mode;
  renderProductGrid();
}

function getFilteredProducts() {
  let list = [...state.catalog];

  // Aisle filter
  if (state.currentAisle !== 'all') {
    list = list.filter(p => p.aisle.toLowerCase() === state.currentAisle.toLowerCase());
  }

  // Search filter
  if (state.searchQuery) {
    const q = state.searchQuery;
    list = list.filter(p =>
      p.name.toLowerCase().includes(q) ||
      p.description.toLowerCase().includes(q) ||
      p.ingredients.toLowerCase().includes(q) ||
      p.tags.toLowerCase().includes(q)
    );
  }

  // Sorting
  if (state.sortMode === 'price-asc') {
    list.sort((a, b) => a.price - b.price);
  } else if (state.sortMode === 'price-desc') {
    list.sort((a, b) => b.price - a.price);
  } else if (state.sortMode === 'rating-desc') {
    list.sort((a, b) => (b.avg_rating || 0) - (a.avg_rating || 0));
  } else if (state.sortMode === 'name-asc') {
    list.sort((a, b) => a.name.localeCompare(b.name));
  }

  return list;
}

function renderProductGrid() {
  const grid = document.getElementById('product-grid');
  const countEl = document.getElementById('displayed-products-count');
  if (!grid) return;

  const products = getFilteredProducts();
  if (countEl) countEl.innerText = products.length;

  if (products.length === 0) {
    grid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 4rem 1rem; color: var(--text-muted);">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔍</div>
        <h3>No matching products found</h3>
        <p>Try searching for marinara, pasta, eggs, tortillas, or reset your aisle filters.</p>
        <button class="btn btn-secondary btn-sm" style="margin-top: 1rem;" onclick="clearSearch(); filterAisle('all');">Reset Filters</button>
      </div>
    `;
    return;
  }

  let html = '';
  for (const p of products) {
    const inCartQty = state.cart[p.id] || 0;
    html += `
      <div class="product-card" onclick="openProductModal('${p.id}')">
        <div class="card-img-box">
          <img src="${p.image_url}" alt="${p.name}" loading="lazy" onerror="this.src='https://images.unsplash.com/photo-1542838132-92c53300491e?w=400';">
          ${p.cold_chain ? '<span class="card-badge-cold">❄️ Cold-Chain</span>' : ''}
          <span class="card-badge-rating">⭐ ${p.avg_rating || '4.8'}</span>
        </div>
        <div class="card-spec">${p.pack_spec} • ${p.aisle}</div>
        <h4 class="card-title">${p.name}</h4>
        <p class="card-desc-snippet">${p.description}</p>
        <div class="card-bottom-row" onclick="event.stopPropagation();">
          <div class="card-price-group">
            <span class="card-price">$${p.price.toFixed(2)}</span>
            <span class="card-calories">${p.calories}</span>
          </div>
          ${inCartQty > 0 ? `
            <div class="cart-qty-stepper">
              <button class="cart-qty-btn" onclick="modifyCartQty('${p.id}', -1)">-</button>
              <span class="cart-qty-num">${inCartQty}</span>
              <button class="cart-qty-btn" onclick="modifyCartQty('${p.id}', 1)">+</button>
            </div>
          ` : `
            <button class="btn-add-cart" onclick="addToCart('${p.id}', 1)">+ Add</button>
          `}
        </div>
      </div>
    `;
  }
  grid.innerHTML = html;
}

// ==========================================
// CART & QUICK-COMMERCE DELIVERY ENGINE
// ==========================================
async function loadCartFromStorageOrDb() {
  const localCartStr = localStorage.getItem('quickbasket_cart');
  if (localCartStr) {
    try {
      state.cart = JSON.parse(localCartStr);
    } catch (e) {
      state.cart = {};
    }
  }

  // If user is authenticated, sync with server
  if (state.user) {
    try {
      const serverCart = await apiFetch('/api/cart');
      if (serverCart.items && serverCart.items.length > 0) {
        // Merge server cart
        for (const itm of serverCart.items) {
          state.cart[itm.product_id] = itm.quantity;
        }
      }
    } catch (e) {
      console.warn("Could not fetch server cart:", e);
    }
  }

  updateCartUI();
}

function persistCart() {
  localStorage.setItem('quickbasket_cart', JSON.stringify(state.cart));
}

async function syncCartWithBackend() {
  if (!state.user) return;
  const items = Object.entries(state.cart).map(([pid, qty]) => ({ product_id: pid, quantity: qty }));
  try {
    await apiFetch('/api/cart/sync', {
      method: 'POST',
      body: JSON.stringify({ items })
    });
  } catch (e) {
    console.warn("Failed syncing cart to DB:", e);
  }
}

function addToCart(productId, qty = 1) {
  state.cart[productId] = (state.cart[productId] || 0) + qty;
  persistCart();
  syncCartWithBackend();
  updateCartUI();
  renderProductGrid();

  const prod = state.catalog.find(p => p.id === productId);
  if (prod) showToast(`Added ${prod.name} to basket`, 'success');
}

function modifyCartQty(productId, delta) {
  const current = state.cart[productId] || 0;
  const next = current + delta;
  if (next <= 0) {
    delete state.cart[productId];
  } else {
    state.cart[productId] = next;
  }
  persistCart();
  syncCartWithBackend();
  updateCartUI();
  renderProductGrid();
}

function toggleCartDrawer() {
  const drawer = document.getElementById('cart-drawer');
  const backdrop = document.getElementById('cart-drawer-backdrop');
  if (!drawer) return;

  const isOpen = drawer.classList.contains('open');
  if (isOpen) {
    drawer.classList.remove('open');
    if (backdrop) backdrop.classList.remove('open');
  } else {
    drawer.classList.add('open');
    if (backdrop) backdrop.classList.add('open');
    refreshCartDiagnostics();
  }
}

async function updateCartUI() {
  const cartEntries = Object.entries(state.cart);
  const totalCount = cartEntries.reduce((sum, [, qty]) => sum + qty, 0);

  // Compute Subtotal
  let subtotal = 0;
  for (const [pid, qty] of cartEntries) {
    const prod = state.catalog.find(p => p.id === pid);
    if (prod) subtotal += prod.price * qty;
  }

  // Header badges
  const headerTotalEl = document.getElementById('header-cart-total');
  const headerBadgeEl = document.getElementById('header-cart-badge');
  const drawerCountEl = document.getElementById('drawer-item-count');

  if (headerTotalEl) headerTotalEl.innerText = `$${subtotal.toFixed(2)}`;
  if (headerBadgeEl) headerBadgeEl.innerText = totalCount;
  if (drawerCountEl) drawerCountEl.innerText = `${totalCount} item${totalCount === 1 ? '' : 's'}`;

  // Drawer Delivery Economics
  const threshold = 15.00;
  const isFreeDelivery = subtotal >= threshold;
  const deliveryFee = isFreeDelivery || totalCount === 0 ? 0.0 : 1.99;
  const needed = Math.max(0, threshold - subtotal);
  const progressPct = Math.min(100, Math.round((subtotal / threshold) * 100));

  const statusText = isFreeDelivery
    ? "🎉 UNLOCKED FREE 10-Minute Delivery!"
    : `Add $${needed.toFixed(2)} more for FREE 10-Minute Delivery`;

  const thresholdStatusEl = document.getElementById('delivery-threshold-status');
  const feeBadgeEl = document.getElementById('delivery-fee-badge');
  const progressFillEl = document.getElementById('delivery-progress-fill');

  if (thresholdStatusEl) thresholdStatusEl.innerText = statusText;
  if (feeBadgeEl) feeBadgeEl.innerText = isFreeDelivery ? "FREE Delivery" : "$1.99 Delivery Fee";
  if (progressFillEl) progressFillEl.style.width = `${progressPct}%`;

  // Drawer Footer Pricing
  const subtotalEl = document.getElementById('drawer-subtotal');
  const delFeeEl = document.getElementById('drawer-delivery-fee');
  const totalEl = document.getElementById('drawer-total-amount');

  if (subtotalEl) subtotalEl.innerText = `$${subtotal.toFixed(2)}`;
  if (delFeeEl) delFeeEl.innerText = `$${deliveryFee.toFixed(2)}`;
  if (totalEl) totalEl.innerText = `$${(subtotal + deliveryFee).toFixed(2)}`;

  // Render items in drawer
  renderCartDrawerItems();

  // Trigger diagnostics if drawer open or items > 0
  if (totalCount > 0) {
    refreshCartDiagnostics();
    fetchLiveCartRecommendations();
  } else {
    const coldBanner = document.getElementById('cold-chain-banner');
    const nudgeBox = document.getElementById('gastronomic-nudge-box');
    const crossSellsWrap = document.getElementById('cart-cross-sells-wrapper');
    const liveBanner = document.getElementById('live-cart-recs-banner');

    if (coldBanner) coldBanner.style.display = 'none';
    if (nudgeBox) nudgeBox.style.display = 'none';
    if (crossSellsWrap) crossSellsWrap.style.display = 'none';
    if (liveBanner) liveBanner.style.display = 'none';
  }
}

function renderCartDrawerItems() {
  const container = document.getElementById('cart-items-container');
  if (!container) return;

  const entries = Object.entries(state.cart);
  if (entries.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; padding: 3rem 1rem; color: var(--text-muted);">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🛍️</div>
        <h4>Your basket is empty</h4>
        <p style="font-size: 0.85rem; margin-top: 0.25rem;">Explore aisles or ask our AI Recipe Copilot for delicious dinner kits!</p>
      </div>
    `;
    return;
  }

  let html = '';
  for (const [pid, qty] of entries) {
    const prod = state.catalog.find(p => p.id === pid);
    if (!prod) continue;
    html += `
      <div class="cart-item-row">
        <div class="cart-item-thumb">
          <img src="${prod.image_url}" alt="${prod.name}">
        </div>
        <div class="cart-item-info">
          <div class="cart-item-title">${prod.name}</div>
          <div class="cart-item-spec">${prod.pack_spec}</div>
          <div class="cart-item-price">$${(prod.price * qty).toFixed(2)}</div>
        </div>
        <div class="cart-qty-stepper">
          <button class="cart-qty-btn" onclick="modifyCartQty('${prod.id}', -1)">-</button>
          <span class="cart-qty-num">${qty}</span>
          <button class="cart-qty-btn" onclick="modifyCartQty('${prod.id}', 1)">+</button>
        </div>
      </div>
    `;
  }
  container.innerHTML = html;
}

async function refreshCartDiagnostics() {
  const cartIds = Object.keys(state.cart);
  if (cartIds.length === 0) return;

  try {
    const diag = await apiFetch('/api/ml/cart-diagnostic', {
      method: 'POST',
      body: JSON.stringify({ cart_items: cartIds })
    });

    // Cold chain banner
    const coldBanner = document.getElementById('cold-chain-banner');
    if (coldBanner) {
      if (diag.cold_chain_compliance && diag.cold_chain_compliance.active) {
        coldBanner.style.display = 'flex';
      } else {
        coldBanner.style.display = 'none';
      }
    }

    // Gastronomic nudges
    const nudgeBox = document.getElementById('gastronomic-nudge-box');
    if (nudgeBox) {
      const nudges = diag.gastronomic_health?.culinary_nudges || [];
      if (nudges.length > 0) {
        nudgeBox.style.display = 'block';
        nudgeBox.innerHTML = `<b>🥘 AI Basket Diagnostic:</b> ${nudges[0]}`;
      } else {
        nudgeBox.style.display = 'none';
      }
    }
  } catch (e) {
    console.warn("Diagnostics error:", e);
  }
}

async function fetchLiveCartRecommendations() {
  const cartIds = Object.keys(state.cart);
  if (cartIds.length === 0) return;

  try {
    const recs = await apiFetch('/api/ml/recommend', {
      method: 'POST',
      body: JSON.stringify({ cart_items: cartIds, top_n: 4, filter_substitutes: true })
    });

    renderBannerRecommendations(recs);
    renderDrawerCrossSells(recs);
  } catch (e) {
    console.warn("Recommendations error:", e);
  }
}

function renderBannerRecommendations(recs) {
  const banner = document.getElementById('live-cart-recs-banner');
  const carousel = document.getElementById('banner-recs-carousel');
  if (!banner || !carousel) return;

  if (!recs || recs.length === 0) {
    banner.style.display = 'none';
    return;
  }

  banner.style.display = 'block';
  let html = '';
  for (const r of recs) {
    html += `
      <div class="recs-chip-card" onclick="addToCart('${r.id}', 1)">
        <img src="${r.image_url}" alt="${r.name}">
        <div class="recs-chip-info">
          <div class="recs-chip-name">${r.name}</div>
          <div class="recs-chip-price">$${r.price.toFixed(2)} • <span style="color:#10b981;">${r.explanation || 'Recommended'}</span></div>
        </div>
        <button class="btn btn-xs btn-primary">+ Add</button>
      </div>
    `;
  }
  carousel.innerHTML = html;
}

function renderDrawerCrossSells(recs) {
  const wrap = document.getElementById('cart-cross-sells-wrapper');
  const list = document.getElementById('cart-cross-sells-list');
  if (!wrap || !list) return;

  if (!recs || recs.length === 0) {
    wrap.style.display = 'none';
    return;
  }

  wrap.style.display = 'block';
  let html = '';
  for (const r of recs) {
    html += `
      <div style="min-width: 130px; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.5rem; text-align: center; cursor: pointer;" onclick="addToCart('${r.id}', 1)">
        <img src="${r.image_url}" style="width: 100%; height: 60px; object-fit: cover; border-radius: 6px; margin-bottom: 4px;">
        <div style="font-size: 0.72rem; font-weight: 700; height: 28px; overflow: hidden; line-height: 1.2;">${r.name}</div>
        <div style="font-size: 0.78rem; font-weight: 800; color: #10b981; margin: 2px 0;">$${r.price.toFixed(2)}</div>
        <button class="btn btn-xs btn-primary btn-block">+ Add</button>
      </div>
    `;
  }
  list.innerHTML = html;
}

// ==========================================
// CHECKOUT & ORDERS MODULE
// ==========================================
function openCheckoutModal() {
  if (Object.keys(state.cart).length === 0) {
    showToast('Your basket is empty. Add items before checking out!', 'info');
    return;
  }
  if (!state.user) {
    openAuthModal('login');
    showToast('Please sign in or use Demo Login to place your order.', 'info');
    return;
  }

  const modal = document.getElementById('checkout-modal');
  if (modal) modal.style.display = 'flex';
  updateCheckoutSummary();
}

function closeCheckoutModal() {
  const modal = document.getElementById('checkout-modal');
  if (modal) modal.style.display = 'none';
}

function applyPromoCode() {
  const code = document.getElementById('checkout-promo').value.trim().toUpperCase();
  const msg = document.getElementById('promo-status-msg');
  if (code === 'AI10') {
    state.appliedPromo = 'AI10';
    if (msg) {
      msg.innerText = '✅ Code AI10 applied: 10% discount!';
      msg.style.color = '#34d399';
    }
    showToast('Promo code AI10 applied!', 'success');
  } else {
    state.appliedPromo = null;
    if (msg) {
      msg.innerText = '❌ Invalid coupon code';
      msg.style.color = '#f87171';
    }
  }
  updateCheckoutSummary();
}

function updateCheckoutSummary() {
  let subtotal = 0;
  for (const [pid, qty] of Object.entries(state.cart)) {
    const prod = state.catalog.find(p => p.id === pid);
    if (prod) subtotal += prod.price * qty;
  }

  const discount = state.appliedPromo === 'AI10' ? subtotal * 0.10 : 0.0;
  const deliveryFee = subtotal >= 15.00 ? 0.0 : 1.99;
  const grandTotal = Math.max(0, subtotal - discount + deliveryFee);

  const subEl = document.getElementById('modal-subtotal');
  const discEl = document.getElementById('modal-discount');
  const feeEl = document.getElementById('modal-delivery-fee');
  const totalEl = document.getElementById('modal-grand-total');

  if (subEl) subEl.innerText = `$${subtotal.toFixed(2)}`;
  if (discEl) discEl.innerText = `-$${discount.toFixed(2)}`;
  if (feeEl) feeEl.innerText = `$${deliveryFee.toFixed(2)}`;
  if (totalEl) totalEl.innerText = `$${grandTotal.toFixed(2)}`;
}

async function executeCheckout(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-submit-order');
  if (btn) btn.disabled = true;

  const address = document.getElementById('checkout-address').value.trim();
  const payment = document.getElementById('checkout-payment').value;
  const promo = state.appliedPromo;

  try {
    const order = await apiFetch('/api/cart/checkout', {
      method: 'POST',
      body: JSON.stringify({
        delivery_address: address,
        payment_method: payment,
        promo_code: promo
      })
    });

    closeCheckoutModal();
    toggleCartDrawer(); // close drawer
    state.cart = {};
    persistCart();
    updateCartUI();

    showToast(`Order #${order.id} confirmed! Arriving in 10 minutes.`, 'success');

    // Refresh user profile & switch to account view
    await initAuth();
    switchTab('account');
  } catch (err) {
    showToast('Checkout failed: ' + err.message, 'error');
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ==========================================
// AI RECIPE & MEAL COPILOT
// ==========================================
function applyRecipePrompt(prompt) {
  const input = document.getElementById('copilot-prompt');
  if (input) input.value = prompt;
  generateMealKit();
}

async function generateMealKit(e) {
  if (e) e.preventDefault();
  const promptInput = document.getElementById('copilot-prompt');
  const servingsSelect = document.getElementById('copilot-servings');
  const budgetSelect = document.getElementById('copilot-budget');
  const btnText = document.getElementById('copilot-btn-text');

  if (!promptInput) return;
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  if (btnText) btnText.innerText = '🧠 NLP Parsing Vector Intent...';

  try {
    const kit = await apiFetch('/api/ml/recipe-copilot', {
      method: 'POST',
      body: JSON.stringify({
        prompt: prompt,
        servings: parseInt(servingsSelect.value || 2),
        max_budget: parseFloat(budgetSelect.value || 35)
      })
    });

    state.currentCopilotKit = kit;
    renderCopilotResult(kit);
  } catch (err) {
    showToast('Failed synthesizing meal kit: ' + err.message, 'error');
  } finally {
    if (btnText) btnText.innerText = '🍳 Synthesize Meal Kit';
  }
}

function renderCopilotResult(kit) {
  const box = document.getElementById('copilot-result-box');
  if (!box || !kit) return;

  box.style.display = 'block';

  let itemsHtml = '';
  for (const itm of kit.items || []) {
    itemsHtml += `
      <div class="kit-item-card">
        <div class="kit-item-img">
          <img src="${itm.image_url}" alt="${itm.product_name}">
        </div>
        <div class="kit-item-role">${itm.role || 'Ingredient'}</div>
        <div class="kit-item-name">${itm.product_name}</div>
        <div class="kit-item-price">$${itm.price.toFixed(2)}</div>
      </div>
    `;
  }

  let stepsHtml = '';
  const steps = kit.recipe_steps || kit.steps || [
    "Prep pantry ingredients and bring a pot of salted water to boil.",
    "Sauté aromatics in extra virgin olive oil over medium flame.",
    "Incorporate sauce and primary proteins until thoroughly simmered.",
    "Toss together, garnish with fresh herbs, and serve hot."
  ];

  steps.forEach((step, idx) => {
    stepsHtml += `
      <div class="recipe-step-item">
        <div class="step-num">${idx + 1}</div>
        <div>${step}</div>
      </div>
    `;
  });

  box.innerHTML = `
    <div class="copilot-result-header">
      <div>
        <span class="cr-theme">${kit.cuisine_theme || 'Chef Special'}</span>
        <h3 class="cr-title">${kit.meal_title || 'Gourmet Artisanal Meal Kit'}</h3>
        <div class="cr-meta">⏱️ Prep Time: <b>${kit.prep_time || '15 mins'}</b> • 🏷️ <b>${kit.dietary_tags || 'Quick-Commerce Balanced'}</b> • Servings: <b>${kit.servings || 2}</b></div>
      </div>
      <div class="cr-match-box">
        <div class="cr-match-lbl">AI Gastronomic Affinity</div>
        <div class="cr-match-val">${kit.ai_confidence || 96}%</div>
      </div>
    </div>

    <div class="cr-narrative-box">
      <b>🧠 AI Culinary Gastronomy Blueprint:</b><br>
      ${kit.culinary_narrative || 'Ingredients harmonized using vector embedding cosine similarity and culinary blueprint logic.'}
    </div>

    <h4 style="margin-bottom: 0.85rem; font-family: Outfit; font-size: 1.15rem;">📦 Assembled Dark Store Pantry &amp; Fresh Ingredients (${(kit.items || []).length} SKUs):</h4>
    <div class="kit-items-grid">
      ${itemsHtml}
    </div>

    <div class="recipe-steps-box">
      <h4>👨‍🍳 15-Minute Chef Cooking Guide:</h4>
      ${stepsHtml}
    </div>

    <div class="copilot-action-bar">
      <div>
        <span style="font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Meal Kit Total:</span>
        <div class="kit-total-price">$${(kit.calculated_total || 14.50).toFixed(2)} <span style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 500;">($${(kit.cost_per_serving || 7.25).toFixed(2)}/serving)</span></div>
      </div>
      <div class="kit-actions-group">
        <button class="btn btn-secondary" onclick="saveCopilotRecipe()">💾 Save Recipe</button>
        <button class="btn btn-primary btn-lg" onclick="loadMealKitToCart()">🛍️ Load Meal Kit into Basket (1-Click)</button>
      </div>
    </div>
  `;

  box.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function loadMealKitToCart() {
  if (!state.currentCopilotKit || !state.currentCopilotKit.items) return;
  for (const itm of state.currentCopilotKit.items) {
    state.cart[itm.product_id] = (state.cart[itm.product_id] || 0) + 1;
  }
  persistCart();
  syncCartWithBackend();
  updateCartUI();
  renderProductGrid();
  toggleCartDrawer();
  showToast(`Loaded ${state.currentCopilotKit.items.length} ingredients into basket!`, 'success');
}

async function saveCopilotRecipe() {
  if (!state.user) {
    openAuthModal('login');
    showToast('Please sign in to save AI recipes to your account.', 'info');
    return;
  }
  if (!state.currentCopilotKit) return;

  try {
    await apiFetch('/api/recipes/save', {
      method: 'POST',
      body: JSON.stringify({
        title: state.currentCopilotKit.meal_title,
        prompt: document.getElementById('copilot-prompt').value,
        cuisine: state.currentCopilotKit.cuisine_theme || 'Gourmet',
        recipe_json: state.currentCopilotKit
      })
    });
    showToast('Recipe saved to your profile!', 'success');
  } catch (e) {
    showToast('Error saving recipe: ' + e.message, 'error');
  }
}

// ==========================================
// 2D SEMANTIC VECTOR MAP (HTML5 CANVAS ENGINE)
// ==========================================
let canvas, ctx;
const AISLE_COLORS = {
  "Pasta & Italian": "#f59e0b",
  "Breakfast & Bakery": "#ec4899",
  "Dairy & Eggs": "#38bdf8",
  "Coffee & Tea": "#d97706",
  "Mexican Fiesta": "#ea580c",
  "Asian Staples": "#10b981",
  "Fresh Produce": "#22c55e",
  "Snacks & Munchies": "#a855f7",
  "Beverages": "#06b6d4",
  "Household & Essentials": "#64748b",
  "General": "#94a3b8"
};

async function loadSemanticMapData() {
  const cartIds = Object.keys(state.cart).join(',');
  try {
    const data = await apiFetch(`/api/ml/semantic-map?cart=${encodeURIComponent(cartIds)}`);
    state.semanticData = data;
    renderCanvasMap();
    renderMapLegend();
    renderCartSimilarityMatrix(data.cart_similarity_matrix);
  } catch (e) {
    console.warn("Error loading semantic map:", e);
  }
}

function initCanvasListeners() {
  canvas = document.getElementById('semantic-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');

  canvas.addEventListener('mousemove', handleCanvasMouseMove);
  canvas.addEventListener('click', handleCanvasClick);
}

function renderMapLegend() {
  const legend = document.getElementById('map-legend');
  if (!legend) return;

  let html = '';
  for (const [aisle, color] of Object.entries(AISLE_COLORS)) {
    html += `
      <div class="legend-item">
        <span class="legend-dot" style="background: ${color};"></span>
        <span>${aisle}</span>
      </div>
    `;
  }
  legend.innerHTML = html;
}

function renderCanvasMap() {
  if (!canvas || !ctx || !state.semanticData) return;

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  // Background grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 50) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  const points = state.semanticData.points || [];
  if (points.length === 0) return;

  // Determine coordinate bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  const padding = 80;
  const scaleX = (width - padding * 2) / (maxX - minX || 1);
  const scaleY = (height - padding * 2) / (maxY - minY || 1);

  // Calculate screen positions
  points.forEach(p => {
    p.screenX = padding + (p.x - minX) * scaleX;
    p.screenY = padding + (p.y - minY) * scaleY;
  });

  // Draw connecting lines between cart items
  const cartPoints = points.filter(p => p.in_cart);
  if (cartPoints.length > 1) {
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 2;
    for (let i = 0; i < cartPoints.length; i++) {
      for (let j = i + 1; j < cartPoints.length; j++) {
        ctx.beginPath();
        ctx.moveTo(cartPoints[i].screenX, cartPoints[i].screenY);
        ctx.lineTo(cartPoints[j].screenX, cartPoints[j].screenY);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  // Draw nodes
  for (const p of points) {
    const isCart = p.in_cart;
    const isHovered = state.hoveredNode && state.hoveredNode.id === p.id;
    const nodeColor = AISLE_COLORS[p.aisle] || '#94a3b8';
    const radius = isCart ? 9 : isHovered ? 8 : 5;

    // Glowing halo for cart items or hovered
    if (isCart || isHovered) {
      ctx.beginPath();
      ctx.arc(p.screenX, p.screenY, radius + 6, 0, Math.PI * 2);
      ctx.fillStyle = isCart ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16, 185, 129, 0.3)';
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(p.screenX, p.screenY, radius, 0, Math.PI * 2);
    ctx.fillStyle = isCart ? '#fbbf24' : nodeColor;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = isCart ? 2 : 1;
    ctx.stroke();

    // Node labels for cart items
    if (isCart) {
      ctx.fillStyle = '#fbbf24';
      ctx.font = 'bold 11px Plus Jakarta Sans';
      ctx.fillText(`★ ${p.name.substring(0, 16)}`, p.screenX + 12, p.screenY + 4);
    }
  }
}

function handleCanvasMouseMove(e) {
  if (!canvas || !state.semanticData) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const points = state.semanticData.points || [];
  let found = null;
  for (const p of points) {
    const dist = Math.hypot(p.screenX - mouseX, p.screenY - mouseY);
    if (dist < 12) {
      found = p;
      break;
    }
  }

  state.hoveredNode = found;
  const tooltip = document.getElementById('map-tooltip');
  if (found && tooltip) {
    tooltip.style.display = 'block';
    tooltip.style.left = `${mouseX + 15}px`;
    tooltip.style.top = `${mouseY + 15}px`;
    tooltip.innerHTML = `
      <div style="font-weight: 800; color: #fff;">${found.name}</div>
      <div style="color: ${AISLE_COLORS[found.aisle] || '#10b981'}; font-size: 0.72rem; text-transform: uppercase;">${found.aisle}</div>
      <div style="color: #10b981; font-weight: 700; margin-top: 2px;">$${found.price.toFixed(2)}</div>
      ${found.in_cart ? '<div style="color: #fbbf24; font-weight: 800; font-size: 0.72rem;">★ IN ACTIVE CART</div>' : ''}
    `;
  } else if (tooltip) {
    tooltip.style.display = 'none';
  }
  renderCanvasMap();
}

function handleCanvasClick() {
  if (state.hoveredNode) {
    openProductModal(state.hoveredNode.id);
  }
}

function resetMapZoom() {
  loadSemanticMapData();
}

function toggleCartSimOverlay() {
  const tableSec = document.getElementById('cart-similarity-section');
  if (tableSec) {
    tableSec.style.display = tableSec.style.display === 'none' ? 'block' : 'none';
  }
}

function renderCartSimilarityMatrix(matrix) {
  const container = document.getElementById('cart-similarity-table-container');
  const section = document.getElementById('cart-similarity-section');
  if (!container || !section) return;

  if (!matrix || matrix.length < 2) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';
  let html = `<table class="bench-table"><thead><tr><th>Product Item</th>`;
  matrix.forEach(row => {
    const prod = state.catalog.find(p => p.id === row[0].source);
    html += `<th>${prod ? prod.name.substring(0, 14) : row[0].source}</th>`;
  });
  html += `</tr></thead><tbody>`;

  matrix.forEach(row => {
    const prod = state.catalog.find(p => p.id === row[0].source);
    html += `<tr><td><b>${prod ? prod.name : row[0].source}</b></td>`;
    row.forEach(cell => {
      const sim = cell.similarity;
      const highlight = sim > 0.4 ? 'color: #10b981; font-weight: 800;' : '';
      html += `<td style="${highlight}">${sim.toFixed(3)}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

// ==========================================
// ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
// ==========================================
function populateABSAPicker() {
  const picker = document.getElementById('absa-sku-picker');
  if (!picker) return;

  let html = '';
  for (const p of state.catalog) {
    html += `<option value="${p.id}">${p.name} (${p.aisle})</option>`;
  }
  picker.innerHTML = html;

  if (state.catalog.length > 0) {
    loadProductABSA(state.catalog[0].id);
  }
}

async function loadProductABSA(productId) {
  try {
    const [prod, reviews] = await Promise.all([
      apiFetch(`/api/catalog/products/${productId}`),
      apiFetch(`/api/reviews/${productId}`)
    ]);

    renderABSAContent(prod, reviews);
  } catch (e) {
    console.warn("ABSA load error:", e);
  }
}

function renderABSAContent(prod, reviews) {
  const area = document.getElementById('absa-content-area');
  if (!area || !prod) return;

  const absa = prod.absa_profile || {
    taste: { positive_pct: 92.0, mention_count: 45 },
    freshness: { positive_pct: 94.5, mention_count: 38 },
    texture: { positive_pct: 89.0, mention_count: 29 },
    value: { positive_pct: 87.5, mention_count: 24 }
  };

  const pairings = prod.culinary_pairings || ["Garlic", "Parmesan", "Basil", "Olive Oil"];

  let reviewsHtml = '';
  for (const r of reviews || []) {
    reviewsHtml += `
      <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.3rem;">
          <b>${r.author_name}</b>
          <span style="color: ${r.sentiment_label === 'POSITIVE' ? '#10b981' : '#f87171'}; font-weight: 700;">${r.sentiment_label} (${(r.sentiment_score).toFixed(2)})</span>
        </div>
        <div style="font-size: 0.84rem; color: var(--text-secondary); line-height: 1.4;">"${r.review_text}"</div>
      </div>
    `;
  }

  area.innerHTML = `
    <div class="absa-overview-grid">
      <div class="absa-prod-card">
        <div class="absa-img-wrap">
          <img src="${prod.image_url}" alt="${prod.name}">
        </div>
        <h3 style="font-family: Outfit; font-size: 1.2rem; margin-bottom: 0.2rem;">${prod.name}</h3>
        <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">${prod.pack_spec} • ${prod.aisle}</div>
        <div style="font-size: 1.3rem; font-weight: 800; color: #fff; margin-bottom: 1rem;">$${prod.price.toFixed(2)}</div>
        <button class="btn btn-primary btn-block" onclick="addToCart('${prod.id}', 1)">+ Add to Basket</button>
      </div>

      <div class="absa-aspects-card">
        <h3 style="font-family: Outfit; font-size: 1.25rem; font-weight: 700; margin-bottom: 0.3rem;">👅 Aspect Sentiment Polarity Breakdown</h3>
        <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 1.5rem;">Calculated via VADER Sentiment Lexicon across customer review tokens.</p>

        <div class="aspect-row">
          <div class="aspect-header">
            <span>👅 Taste &amp; Flavor Profile</span>
            <span style="color: #10b981;">${absa.taste.positive_pct}% Positive (${absa.taste.mention_count} mentions)</span>
          </div>
          <div class="aspect-bar-track"><div class="aspect-bar-fill" style="width: ${absa.taste.positive_pct}%;"></div></div>
        </div>

        <div class="aspect-row">
          <div class="aspect-header">
            <span>🌿 Freshness &amp; Purity</span>
            <span style="color: #10b981;">${absa.freshness.positive_pct}% Positive (${absa.freshness.mention_count} mentions)</span>
          </div>
          <div class="aspect-bar-track"><div class="aspect-bar-fill" style="width: ${absa.freshness.positive_pct}%;"></div></div>
        </div>

        <div class="aspect-row">
          <div class="aspect-header">
            <span>📦 Texture &amp; Consistency</span>
            <span style="color: #10b981;">${absa.texture.positive_pct}% Positive (${absa.texture.mention_count} mentions)</span>
          </div>
          <div class="aspect-bar-track"><div class="aspect-bar-fill" style="width: ${absa.texture.positive_pct}%;"></div></div>
        </div>

        <div class="aspect-row">
          <div class="aspect-header">
            <span>💰 Value for Money</span>
            <span style="color: #10b981;">${absa.value.positive_pct}% Positive (${absa.value.mention_count} mentions)</span>
          </div>
          <div class="aspect-bar-track"><div class="aspect-bar-fill" style="width: ${absa.value.positive_pct}%;"></div></div>
        </div>

        <h4 style="margin: 1.5rem 0 0.5rem 0; font-size: 0.95rem;">🌿 Mined Co-Mention Pairings:</h4>
        <div class="pairings-cloud">
          ${pairings.map(tag => `<span class="pairing-tag">🥗 ${tag}</span>`).join('')}
        </div>

        <h4 style="margin: 1.5rem 0 0.75rem 0; font-size: 0.95rem;">💬 Verified Shopper Review Stream:</h4>
        <div class="reviews-stream">${reviewsHtml}</div>
      </div>
    </div>
  `;
}

async function submitReview(e) {
  e.preventDefault();
  const picker = document.getElementById('absa-sku-picker');
  const authorInput = document.getElementById('review-author');
  const ratingSelect = document.getElementById('review-rating');
  const textInput = document.getElementById('review-text');

  if (!picker || !textInput) return;
  const productId = picker.value;
  const author = authorInput.value.trim() || (state.user ? state.user.full_name : 'Verified Shopper');
  const rating = parseInt(ratingSelect.value || 5);
  const text = textInput.value.trim();

  try {
    const rev = await apiFetch('/api/reviews', {
      method: 'POST',
      body: JSON.stringify({
        product_id: productId,
        author_name: author,
        rating: rating,
        review_text: text
      })
    });

    showToast(`Review published! VADER Polarity: ${rev.sentiment_label} (${rev.sentiment_score.toFixed(2)})`, 'success');
    textInput.value = '';
    loadProductABSA(productId);
  } catch (err) {
    showToast('Failed submitting review: ' + err.message, 'error');
  }
}

// ==========================================
// SMART PROMOTIONAL BUNDLES MODULE
// ==========================================
async function loadBundlesData() {
  try {
    const bundles = await apiFetch('/api/ml/bundles');
    renderBundles(bundles);
  } catch (e) {
    console.warn("Bundles load error:", e);
  }
}

function renderBundles(bundles) {
  const grid = document.getElementById('bundles-grid');
  if (!grid || !bundles) return;

  let html = '';
  for (const b of bundles) {
    const items = b.items || [];
    const lift = b.lift || 4.2;
    const origPrice = b.original_total_price || (b.bundle_price * 1.18);
    const currPrice = b.bundle_price || 12.50;
    const saveAmt = origPrice - currPrice;

    let itemsHtml = '';
    items.forEach(itm => {
      itemsHtml += `
        <div class="bundle-item-thumb">
          <div class="bundle-thumb-img">
            <img src="${itm.image_url}" alt="${itm.name}">
          </div>
          <div class="bundle-item-name">${itm.name}</div>
        </div>
      `;
    });

    const itemIds = items.map(i => i.product_id).join(',');

    html += `
      <div class="bundle-card">
        <span class="bundle-lift-badge">⚡ ${lift.toFixed(1)}x Lift</span>
        <h3 class="bundle-title">${b.bundle_name || 'Curated Culinary Kit'}</h3>
        <div class="bundle-theme">🌿 ${b.theme || 'Quick-Commerce Bestseller'}</div>
        <div class="bundle-items-row">${itemsHtml}</div>
        <div class="bundle-pricing-row">
          <div class="bundle-prices">
            <span class="bundle-price-curr">$${currPrice.toFixed(2)}</span>
            <span class="bundle-price-orig">$${origPrice.toFixed(2)}</span>
            <span class="bundle-save-pill">SAVE $${saveAmt.toFixed(2)}</span>
          </div>
          <button class="btn btn-primary btn-sm" onclick="addBundleToCart('${itemIds}')">🛍️ Add Bundle</button>
        </div>
      </div>
    `;
  }
  grid.innerHTML = html;
}

function addBundleToCart(idsStr) {
  const ids = idsStr.split(',').filter(Boolean);
  ids.forEach(id => {
    state.cart[id] = (state.cart[id] || 0) + 1;
  });
  persistCart();
  syncCartWithBackend();
  updateCartUI();
  renderProductGrid();
  toggleCartDrawer();
  showToast(`Added entire bundle (${ids.length} items) to basket!`, 'success');
}

// ==========================================
// COLD-START SOLVER STUDIO
// ==========================================
function fillColdStart(name, desc, ingr, aisle) {
  document.getElementById('cs-name').value = name;
  document.getElementById('cs-desc').value = desc;
  document.getElementById('cs-ingr').value = ingr;
  document.getElementById('cs-aisle').value = aisle;
}

async function solveColdStart(e) {
  e.preventDefault();
  const name = document.getElementById('cs-name').value.trim();
  const desc = document.getElementById('cs-desc').value.trim();
  const ingr = document.getElementById('cs-ingr').value.trim();
  const aisle = document.getElementById('cs-aisle').value;
  const container = document.getElementById('cs-results-content');

  if (!container) return;
  container.innerHTML = `<div class="placeholder-box">🧠 Vectorizing novel SKU across 2,500-term TF-IDF space and computing cosine distance...</div>`;

  try {
    const res = await apiFetch('/api/ml/cold-start', {
      method: 'POST',
      body: JSON.stringify({ name, description: desc, ingredients: ingr, aisle })
    });

    let recsHtml = '';
    for (const r of res.recommendations || []) {
      recsHtml += `
        <div style="display: flex; gap: 0.85rem; align-items: center; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.6rem;">
          <img src="${r.image_url}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 6px;">
          <div style="flex: 1;">
            <div style="font-weight: 700; font-size: 0.85rem; color: #fff;">${r.name}</div>
            <div style="font-size: 0.74rem; color: #10b981; font-weight: 700;">Cosine Similarity: ${(r.cosine_similarity).toFixed(3)} • ${r.aisle}</div>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">${r.rationale}</div>
          </div>
          <button class="btn btn-xs btn-primary" onclick="addToCart('${r.id}', 1)">+ Add</button>
        </div>
      `;
    }

    container.innerHTML = `
      <div class="cs-result-banner">
        <b>✅ Cold-Start Problem Overcome!</b><br>
        Traditional FP-Growth produces <b>0 rules</b> because this item has 0 transactions. The NLP Semantic Engine vectorized the recipe description &amp; ingredients into 2,500-term TF-IDF coordinates to compute instant cross-sells.
      </div>
      <div>${recsHtml}</div>
    `;
  } catch (err) {
    container.innerHTML = `<div class="placeholder-box" style="color: #f87171;">Error solving cold start: ${err.message}</div>`;
  }
}

// ==========================================
// ACCOUNT DASHBOARD & ORDERS
// ==========================================
async function updateAccountView() {
  const loggedOutCard = document.getElementById('account-logged-out');
  const loggedInCard = document.getElementById('account-logged-in');

  if (!loggedOutCard || !loggedInCard) return;

  if (!state.user) {
    loggedOutCard.style.display = 'block';
    loggedInCard.style.display = 'none';
  } else {
    loggedOutCard.style.display = 'none';
    loggedInCard.style.display = 'block';

    const nameEl = document.getElementById('user-display-name');
    const emailEl = document.getElementById('user-display-email');
    const pointsEl = document.getElementById('user-loyalty-pts');
    const avatarEl = document.getElementById('user-avatar-img');

    if (nameEl) nameEl.innerText = state.user.full_name;
    if (emailEl) emailEl.innerText = state.user.email;
    if (pointsEl) pointsEl.innerText = state.user.loyalty_points;
    if (avatarEl && state.user.avatar_url) avatarEl.src = state.user.avatar_url;

    await loadUserOrdersAndRecipes();
  }
}

async function loadUserOrdersAndRecipes() {
  if (!state.user) return;
  try {
    const [orders, recipes] = await Promise.all([
      apiFetch('/api/orders'),
      apiFetch('/api/recipes/saved')
    ]);

    renderOrdersList(orders);
    renderSavedRecipesList(recipes);
  } catch (e) {
    console.warn("Could not load account history:", e);
  }
}

function renderOrdersList(orders) {
  const container = document.getElementById('orders-history-list');
  const countBadge = document.getElementById('orders-count-badge');
  if (!container) return;

  if (countBadge) countBadge.innerText = `${orders.length} order${orders.length === 1 ? '' : 's'}`;

  if (!orders || orders.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem; padding: 1.5rem; text-align: center;">No orders yet. Place your first 10-minute order!</div>`;
    return;
  }

  let html = '';
  for (const o of orders) {
    const items = o.items || [];
    const itemNames = items.map(i => `${i.name} (x${i.quantity || 1})`).join(', ');

    html += `
      <div class="order-card">
        <div class="order-top">
          <span class="order-id">#${o.id}</span>
          <span class="order-status-badge">⚡ ${o.status}</span>
        </div>
        <div class="order-items-summary">${itemNames}</div>
        <div class="order-bottom">
          <span>${o.cold_chain ? '❄️ Cold-Chain Pouch Included • ' : ''}${o.estimated_arrival_minutes}m SLA</span>
          <span class="order-total">$${o.total_amount.toFixed(2)}</span>
        </div>
      </div>
    `;
  }
  container.innerHTML = html;
}

function renderSavedRecipesList(recipes) {
  const container = document.getElementById('saved-recipes-list');
  const countBadge = document.getElementById('recipes-count-badge');
  if (!container) return;

  if (countBadge) countBadge.innerText = `${recipes.length} saved`;

  if (!recipes || recipes.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem; padding: 1.5rem; text-align: center;">No saved recipes yet. Ask Chef AI in the Recipe Copilot tab!</div>`;
    return;
  }

  let html = '';
  for (const r of recipes) {
    html += `
      <div style="background: var(--bg-surface-card); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;">
        <div style="font-size: 0.72rem; font-weight: 800; color: #10b981; text-transform: uppercase;">${r.cuisine}</div>
        <div style="font-family: Outfit; font-weight: 700; font-size: 1rem; color: #fff; margin: 2px 0 4px 0;">${r.title}</div>
        <div style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.6rem;">"${r.prompt}"</div>
        <button class="btn btn-xs btn-outline" onclick="applyRecipePrompt('${r.prompt.replace(/'/g, "\\'")}'); switchTab('copilot');">Cook Again 🍳</button>
      </div>
    `;
  }
  container.innerHTML = html;
}

// ==========================================
// PRODUCT DETAIL MODAL
// ==========================================
async function openProductModal(productId) {
  state.activeProductModalId = productId;
  const modal = document.getElementById('product-modal');
  const body = document.getElementById('product-modal-body');
  const aisleEl = document.getElementById('modal-prod-aisle');

  if (modal) modal.style.display = 'flex';
  if (body) body.innerHTML = '<div style="text-align: center; padding: 2rem;">Loading details...</div>';

  try {
    const [prod, reviews] = await Promise.all([
      apiFetch(`/api/catalog/products/${productId}`),
      apiFetch(`/api/reviews/${productId}`)
    ]);

    if (aisleEl) aisleEl.innerText = prod.aisle;
    const inCartQty = state.cart[prod.id] || 0;

    let similarHtml = '';
    for (const s of prod.similar_products || []) {
      similarHtml += `
        <div style="text-align: center; cursor: pointer;" onclick="openProductModal('${s.id}')">
          <img src="${s.image_url}" style="width: 100%; height: 70px; object-fit: cover; border-radius: 8px; margin-bottom: 4px;">
          <div style="font-size: 0.72rem; font-weight: 700; height: 28px; overflow: hidden; line-height: 1.2;">${s.name}</div>
          <div style="font-size: 0.78rem; color: #10b981; font-weight: 800;">$${s.price.toFixed(2)}</div>
        </div>
      `;
    }

    body.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1.2fr; gap: 1.5rem; margin-bottom: 1.5rem;">
        <div>
          <img src="${prod.image_url}" style="width: 100%; height: 240px; object-fit: cover; border-radius: 12px; margin-bottom: 0.75rem;">
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted);">
            <span>Pack: <b>${prod.pack_spec}</b></span>
            <span>Energy: <b>${prod.calories}</b></span>
          </div>
        </div>
        <div>
          <h2 style="font-family: Outfit; font-size: 1.4rem; font-weight: 800; margin-bottom: 0.4rem;">${prod.name}</h2>
          <div style="display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.75rem;">
            <span style="font-family: Outfit; font-size: 1.6rem; font-weight: 800; color: #fff;">$${prod.price.toFixed(2)}</span>
            <span style="color: #fbbf24; font-size: 0.85rem; font-weight: 700;">⭐ ${prod.avg_rating} (${prod.review_count} ratings)</span>
          </div>
          <p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 0.85rem;">${prod.description}</p>
          <div style="background: var(--bg-surface); padding: 0.75rem; border-radius: 8px; font-size: 0.78rem; color: var(--text-muted); margin-bottom: 1.25rem;">
            <b>Ingredients:</b> ${prod.ingredients}
          </div>
          <button class="btn btn-primary btn-block btn-lg" onclick="addToCart('${prod.id}', 1); closeProductModal();">
            ${inCartQty > 0 ? `+ Add More (Current: ${inCartQty})` : '+ Add to Basket ($' + prod.price.toFixed(2) + ')'}
          </button>
        </div>
      </div>

      <div style="border-top: 1px solid var(--border-subtle); padding-top: 1rem;">
        <h4 style="font-size: 0.92rem; font-weight: 700; margin-bottom: 0.75rem;">🌿 NLP Semantic Complements (Ingredient Overlap):</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 0.75rem;">
          ${similarHtml}
        </div>
      </div>
    `;
  } catch (e) {
    body.innerHTML = `<div style="color: #f87171;">Failed to load product details: ${e.message}</div>`;
  }
}

function closeProductModal() {
  const modal = document.getElementById('product-modal');
  if (modal) modal.style.display = 'none';
}

// ==========================================
// TAB ROUTING
// ==========================================
function switchTab(tabId) {
  state.activeTab = tabId;

  // Nav buttons
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });

  // Tab views
  document.querySelectorAll('.tab-view').forEach(view => {
    view.classList.toggle('active', view.id === `view-${tabId}`);
  });

  // Lazy tab data loaders
  if (tabId === 'semantic-map') {
    loadSemanticMapData();
  } else if (tabId === 'bundles') {
    loadBundlesData();
  } else if (tabId === 'account') {
    updateAccountView();
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}
