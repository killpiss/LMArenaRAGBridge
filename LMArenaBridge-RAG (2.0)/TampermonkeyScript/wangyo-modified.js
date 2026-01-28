// ==UserScript==
// @name         LMArena Auto Refresh
// @version      69 mwehehhe
// @description  Automates refreshing and other interactions on LMArena, with summarizer compatibility.
// @author       Gemini best vibecode buddy
// @match        https://*.lmarena.ai/*
// @run-at       document-start
// @all-frames   true
// ==/UserScript==

(function () {
'use strict';

// ================================================================= //
// ==================== CONFIGURATION ====================
// ================================================================= //
const CONFIG = {
    SERVER_URL: "ws://localhost:9080/ws",
    REQUIRED_COOKIE: "arena-auth-prod-v1",
    TURNSTILE_SITEKEY: '0x4AAAAAAA65vWDmG-O_lPtT',
    MAX_REQUESTS_PER_TAB: 999,  // ✅ CHANGED: Effectively disabled - only rotate on 429
    DEBUG: false  // Set to true for more verbose logging
};

// ================================================================= //
// ==================== LOGGING UTILITIES (MOVED UP!) ====================
// ================================================================= //
const log = {
    info: (...args) => console.log('[Stealth Bridge]', ...args),
    warn: (...args) => console.warn('[Stealth Bridge]', ...args),
    error: (...args) => console.error('[Stealth Bridge]', ...args),
    debug: (...args) => CONFIG.DEBUG && console.log('[DEBUG]', ...args),
    success: (msg) => console.log('%c' + msg, 'color: #28a745; font-weight: bold;')
};

// ================================================================= //
// ==================== ANTI-DETECTION SETUP ====================
// ================================================================= //

Object.defineProperty(navigator, 'webdriver', {
    get: () => false
});

window.chrome = window.chrome || {};
window.chrome.runtime = window.chrome.runtime || {};

(function() {
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {
        if (parameter === 37445) {
            const vendors = ['Intel Inc.', 'NVIDIA Corporation', 'AMD', 'Apple Inc.'];
            return vendors[Math.floor(Math.random() * vendors.length)];
        }
        if (parameter === 37446) {
            const renderers = [
                'Intel Iris OpenGL Engine',
                'NVIDIA GeForce GTX 1060',
                'AMD Radeon Pro 580',
                'Apple M1'
            ];
            return renderers[Math.floor(Math.random() * renderers.length)];
        }
        return getParameter.call(this, parameter);
    };
})();

(function() {
    const toDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function(type) {
        if (type === 'image/png' && this.width === 280 && this.height === 60) {
            const noise = Math.random() * 0.001;
            const ctx = this.getContext('2d');
            const imageData = ctx.getImageData(0, 0, this.width, this.height);
            for (let i = 0; i < imageData.data.length; i += 4) {
                imageData.data[i] = imageData.data[i] + noise;
            }
            ctx.putImageData(imageData, 0, 0);
        }
        return toDataURL.apply(this, arguments);
    };
})();

(function() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) {
        const getChannelData = AudioBuffer.prototype.getChannelData;
        AudioBuffer.prototype.getChannelData = function(channel) {
            const data = getChannelData.call(this, channel);
            for (let i = 0; i < data.length; i += 100) {
                data[i] = data[i] + Math.random() * 0.0001;
            }
            return data;
        };
    }
})();

// ================================================================= //
// ==================== MOUSE MOVEMENT SIMULATION ====================
// ================================================================= //

class MouseSimulator {
    constructor() {
        this.isActive = false;
        this.lastMove = Date.now();
    }

    bSpline(points, t) {
        const n = points.length - 1;
        if (n < 3) return points[Math.min(Math.floor(t * n), n - 1)];

        const segment = Math.min(Math.floor(t * (n - 2)), n - 3);
        const localT = (t * (n - 2)) - segment;

        const p0 = points[segment];
        const p1 = points[segment + 1];
        const p2 = points[segment + 2];
        const p3 = points[segment + 3];

        const t2 = localT * localT;
        const t3 = t2 * localT;

        const x = (
            p0.x * (-t3 + 3*t2 - 3*localT + 1) +
            p1.x * (3*t3 - 6*t2 + 4) +
            p2.x * (-3*t3 + 3*t2 + 3*localT + 1) +
            p3.x * t3
        ) / 6;

        const y = (
            p0.y * (-t3 + 3*t2 - 3*localT + 1) +
            p1.y * (3*t3 - 6*t2 + 4) +
            p2.y * (-3*t3 + 3*t2 + 3*localT + 1) +
            p3.y * t3
        ) / 6;

        return { x, y };
    }

    generatePath(startX, startY, endX, endY) {
        const points = [{ x: startX, y: startY }];
        const steps = 3 + Math.floor(Math.random() * 3);

        for (let i = 1; i < steps; i++) {
            const progress = i / steps;
            const x = startX + (endX - startX) * progress + (Math.random() - 0.5) * 100;
            const y = startY + (endY - startY) * progress + (Math.random() - 0.5) * 100;
            points.push({ x, y });
        }

        points.push({ x: endX, y: endY });
        return points;
    }

    async moveMouse(targetX, targetY) {
        const startX = window.innerWidth / 2;
        const startY = window.innerHeight / 2;

        const path = this.generatePath(startX, startY, targetX, targetY);
        const duration = 300 + Math.random() * 200;
        const steps = 20;

        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const point = this.bSpline(path, t);

            const element = document.elementFromPoint(point.x, point.y) || document.body;

            element.dispatchEvent(new MouseEvent('mousemove', {
                bubbles: true,
                cancelable: true,
                clientX: point.x,
                clientY: point.y,
                view: window
            }));

            await new Promise(resolve => setTimeout(resolve, duration / steps));
        }

        this.lastMove = Date.now();
    }

    startRandomMovements() {
        if (this.isActive) return;
        this.isActive = true;

        const moveInterval = () => {
            if (!this.isActive) return;

            const x = Math.random() * window.innerWidth;
            const y = Math.random() * window.innerHeight;

            this.moveMouse(x, y);

            const nextMove = 5000 + Math.random() * 10000;
            setTimeout(moveInterval, nextMove);
        };

        moveInterval();
    }

    stop() {
        this.isActive = false;
    }
}

const mouseSimulator = new MouseSimulator();

// ================================================================= //
// ==================== TAB ROTATION MANAGER ====================
// ================================================================= //

class TabRotationManager {
    constructor() {
        this.requestCount = parseInt(localStorage.getItem('lmarena_requestCount') || '0');
        this.tabId = localStorage.getItem('lmarena_tabId') || this.generateTabId();
        this.shouldRotate = false;

        if (!localStorage.getItem('lmarena_tabId')) {
            localStorage.setItem('lmarena_tabId', this.tabId);
        }

        log.info(`📋 Tab ID: ${this.tabId} | Requests: ${this.requestCount}`);
    }

    generateTabId() {
        return 'tab_' + Math.random().toString(36).substring(2, 15);
    }

    incrementRequest() {
        this.requestCount++;
        localStorage.setItem('lmarena_requestCount', this.requestCount.toString());

        log.debug(`📊 Request count: ${this.requestCount}/${CONFIG.MAX_REQUESTS_PER_TAB}`);

        // This will never trigger with MAX_REQUESTS_PER_TAB = 999
        // Rotation only happens on 429 errors
        if (this.requestCount >= CONFIG.MAX_REQUESTS_PER_TAB) {
            this.shouldRotate = true;
            log.info('🔄 Tab rotation threshold reached');
        }
    }

    // ✅ ENHANCED: Clear auth and create fresh identity
    resetAndRotate() {
        log.info('🔄 Rotating to new tab with FRESH identity...');

        // Clear ALL auth data to force new identity
        log.info('🗑️ Clearing all auth data...');

        // Clear localStorage
        localStorage.removeItem('lmarena_auth_data');
        localStorage.removeItem('lmarena_auth_timestamp');
        localStorage.setItem('lmarena_requestCount', '0');
        localStorage.setItem('lmarena_tabId', this.generateTabId());
        localStorage.setItem('lmarena_needs_fresh_identity', 'true');

        // Clear all cookies
        const cookies = document.cookie.split(";");
        for (let cookie of cookies) {
            const name = cookie.split("=")[0].trim();
            if (name) {
                document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
                document.cookie = `${name}=; path=/; domain=lmarena.ai; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
                document.cookie = `${name}=; path=/; domain=.lmarena.ai; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
            }
        }

        log.success('✅ Auth cleared, opening fresh tab...');

        // Open new tab with fresh identity
        window.open(window.location.href, '_blank');

        // Close this tab after brief delay
        setTimeout(() => {
            log.info('🚪 Closing old tab...');
            window.close();
        }, 2000);
    }

    shouldRotateNow() {
        return this.shouldRotate;
    }
}

const tabManager = new TabRotationManager();

// ================================================================= //
// ==================== STATE MANAGEMENT ====================
// ================================================================= //
let socket;
let modelRegistrySent = false;
let latestTurnstileToken = null;
const activeFetchControllers = new Map();

// ================================================================= //
// ==================== MODEL EXTRACTION ====================
// ================================================================= //
function extractModelRegistry() {
    log.info('🔍 Starting model extraction...');

    try {
        const scripts = document.querySelectorAll('script');
        log.debug(`Found ${scripts.length} script tags`);

        let modelData = null;

        for (const script of scripts) {
            const content = script.textContent || script.innerHTML;

            const regex = /self\.__next_f\.push\(\[(\d+),"([^"]+?):(.*?)"\]\)/g;
            let match;

            while ((match = regex.exec(content)) !== null) {
                const moduleDataStr = match[3];

                if (moduleDataStr.includes('initialModels') || moduleDataStr.includes('initialState')) {
                    try {
                        const unescapedData = moduleDataStr
                            .replace(/\\"/g, '"')
                            .replace(/\\\\/g, '\\')
                            .replace(/\\n/g, '\n')
                            .replace(/\\r/g, '\r')
                            .replace(/\\t/g, '\t');

                        const parsedData = JSON.parse(unescapedData);
                        modelData = findModelsRecursively(parsedData);

                        if (modelData && modelData.length > 0) {
                            log.success(`✅ Found ${modelData.length} models`);
                            break;
                        }
                    } catch (parseError) {
                        try {
                            const bracketMatch = moduleDataStr.match(/(\[.*\])/);
                            if (bracketMatch) {
                                const bracketData = bracketMatch[1]
                                    .replace(/\\"/g, '"')
                                    .replace(/\\\\/g, '\\');
                                const parsedBracketData = JSON.parse(bracketData);
                                modelData = findModelsRecursively(parsedBracketData);

                                if (modelData && modelData.length > 0) {
                                    log.success(`✅ Found ${modelData.length} models (bracket)`);
                                    break;
                                }
                            }
                        } catch (altError) {
                            // Continue
                        }
                    }
                }
            }

            if (modelData) break;
        }

        if (!modelData || modelData.length === 0) {
            log.warn('⚠️ No models found, will retry...');
            return null;
        }

        const registry = {};

        modelData.forEach(model => {
            if (!model || typeof model !== 'object' || !model.publicName) return;
            if (registry[model.publicName]) return;

            let type = 'chat';
            if (model.capabilities && model.capabilities.outputCapabilities) {
                if (model.capabilities.outputCapabilities.image) type = 'image';
                else if (model.capabilities.outputCapabilities.video) type = 'video';
            }

            registry[model.publicName] = {
                type: type,
                ...model
            };
        });

        log.success(`✅ Registry built: ${Object.keys(registry).length} models`);

        return registry;

    } catch (error) {
        log.error('❌ Extraction error:', error);
        return null;
    }
}

function findModelsRecursively(obj, depth = 0) {
    if (depth > 15) return null;
    if (!obj || typeof obj !== 'object') return null;

    const modelKeys = ['initialModels', 'initialState', 'models', 'modelList'];

    for (const key of modelKeys) {
        if (obj[key] && Array.isArray(obj[key])) {
            const models = obj[key];

            if (models.length > 0 && models[0] &&
                (models[0].publicName || models[0].name || models[0].id)) {
                return models;
            }
        }
    }

    if (Array.isArray(obj)) {
        for (const item of obj) {
            const result = findModelsRecursively(item, depth + 1);
            if (result) return result;
        }
    }

    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            const result = findModelsRecursively(obj[key], depth + 1);
            if (result) return result;
        }
    }

    return null;
}

// ================================================================= //
// ==================== TURNSTILE ====================
// ================================================================= //

const originalCreateElement = document.createElement;
document.createElement = function(...args) {
    const element = originalCreateElement.apply(this, args);

    if (element.tagName === 'SCRIPT') {
        const originalSetAttribute = element.setAttribute;
        element.setAttribute = function(name, value) {
            originalSetAttribute.call(this, name, value);

            if (name === 'src' && value && value.includes('challenges.cloudflare.com/turnstile')) {
                element.addEventListener('load', function() {
                    if (window.turnstile) {
                        hookTurnstileRender(window.turnstile);
                    }
                });

                document.createElement = originalCreateElement;
            }
        };
    }
    return element;
};

function hookTurnstileRender(turnstile) {
    const originalRender = turnstile.render;
    turnstile.render = function(container, params) {
        const originalCallback = params.callback;
        params.callback = (token) => {
            latestTurnstileToken = token;
            log.success(`✅ Token: ${token.substring(0, 20)}...`);
            if (originalCallback) return originalCallback(token);
        };
        return originalRender(container, params);
    };
}

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

function checkAuthCookie() {
    return !!getCookie(CONFIG.REQUIRED_COOKIE);
}

// ================================================================= //
// ==================== FETCH ====================
// ================================================================= //
async function executeFetchAndStreamBack(requestId, payload) {
    const abortController = new AbortController();
    activeFetchControllers.set(requestId, abortController);

    try {
        const response = await fetch('/nextjs-api/stream/create-evaluation', {
            method: 'POST',
            headers: {
                'Content-Type': 'text/plain;charset=UTF-8',
                'Accept': '*/*',
            },
            body: JSON.stringify(payload),
            signal: abortController.signal
        });

        // ✅ ENHANCED: Handle 429 rate limit with fresh identity
        if (response.status === 429) {
            log.warn(`🚫 Rate limit (429) detected! Rotating to fresh identity...`);

            // Clear all auth
            const cookies = document.cookie.split(";");
            for (let cookie of cookies) {
                const name = cookie.split("=")[0].trim();
                if (name) {
                    document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
                    document.cookie = `${name}=; path=/; domain=lmarena.ai; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
                    document.cookie = `${name}=; path=/; domain=.lmarena.ai; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
                }
            }

            localStorage.removeItem('lmarena_auth_data');
            localStorage.removeItem('lmarena_auth_timestamp');
            localStorage.setItem('lmarena_requestCount', '0');
            localStorage.setItem('lmarena_tabId', tabManager.generateTabId());

            log.info('🔄 Refreshing page with fresh identity...');
            window.location.reload();
            return;
        }

        if (!response.ok || !response.body) {
            throw new Error(`Fetch failed: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                sendToServer(requestId, "[DONE]");

                // Increment request count (though it won't trigger rotation at 999)
                tabManager.incrementRequest();
                if (tabManager.shouldRotateNow()) {
                    setTimeout(() => tabManager.resetAndRotate(), 2000);
                }
                break;
            }

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());
            for (const line of lines) {
                sendToServer(requestId, line);
            }
        }

    } catch (error) {
        if (error.name !== 'AbortError') {
            log.error(`Fetch error:`, error);
            sendToServer(requestId, JSON.stringify({ error: error.message }));
            sendToServer(requestId, "[DONE]");
        }
    } finally {
        activeFetchControllers.delete(requestId);
    }
}

// ================================================================= //
// ==================== WEBSOCKET ====================
// ================================================================= //
function connect() {
    log.info(`Connecting to ${CONFIG.SERVER_URL}...`);
    socket = new WebSocket(CONFIG.SERVER_URL);

    socket.onopen = () => {
        log.success("✅ Connected");

        socket.send(JSON.stringify({
            type: 'reconnection_handshake',
            tab_id: tabManager.tabId,
            request_count: tabManager.requestCount,
            timestamp: Date.now()
        }));

        if (!modelRegistrySent) {
            setTimeout(sendModelRegistry, 2000);
        }

        mouseSimulator.startRandomMovements();
    };

    socket.onmessage = async (event) => {
        try {
            const message = JSON.parse(event.data);

            if (message.type === 'model_registry_ack') {
                log.success(`✅ Server got ${message.count} models`);
                modelRegistrySent = true;
                return;
            }

            if (message.type === 'abort_request') {
                const controller = activeFetchControllers.get(message.request_id);
                if (controller) {
                    controller.abort();
                    activeFetchControllers.delete(message.request_id);
                }
                return;
            }

            const { request_id, payload } = message;

            if (request_id && payload) {
                await executeFetchAndStreamBack(request_id, payload);
            }

        } catch (error) {
            log.error("Message error:", error);
        }
    };

    socket.onclose = () => {
        log.warn("🔌 Disconnected. Retrying in 5s...");
        modelRegistrySent = false;
        mouseSimulator.stop();
        setTimeout(connect, 5000);
    };

    socket.onerror = (error) => {
        log.error("WS error:", error);
        socket.close();
    };
}

function sendToServer(requestId, data) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            request_id: requestId,
            data: data
        }));
    }
}

function sendModelRegistry() {
    log.info('📤 Sending models...');

    if (!socket || socket.readyState !== WebSocket.OPEN) {
        log.warn('⚠️ WebSocket not ready');
        return;
    }

    const models = extractModelRegistry();

    if (!models || Object.keys(models).length === 0) {
        log.warn('⚠️ No models, retrying in 5s');
        setTimeout(sendModelRegistry, 5000);
        return;
    }

    socket.send(JSON.stringify({
        type: 'model_registry',
        models: models
    }));

    log.success(`📤 Sent ${Object.keys(models).length} models`);
}

// ================================================================= //
// ==================== INIT ====================
// ================================================================= //
log.success('🚀 Bridge V4.0 Starting...');
log.info('🛡️ Smart rotation: Only on rate limit (429)');
log.info('⏱️ Server-side delays: 2-5 seconds between requests');

connect();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(sendModelRegistry, 3000);
    });
} else {
    setTimeout(sendModelRegistry, 3000);
}

log.success('✅ Initialized!');

})();