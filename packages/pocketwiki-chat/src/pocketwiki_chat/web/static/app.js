/**
 * PocketWiki Chat Application
 */

const elements = {
    form: document.getElementById('chat-form'),
    input: document.getElementById('query-input'),
    sendBtn: document.getElementById('send-btn'),
    messages: document.getElementById('messages'),
    sourcesList: document.getElementById('sources-list'),
};

let isStreaming = false;
let currentSources = [];

/**
 * Add a message to the chat
 */
function addMessage(role, content, isLoading = false) {
    // Remove welcome message if present
    const welcome = elements.messages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    if (isLoading) msg.classList.add('loading');

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.textContent = content;
    msg.appendChild(contentDiv);

    elements.messages.appendChild(msg);
    elements.messages.scrollTop = elements.messages.scrollHeight;

    return msg;
}

/**
 * Update the last assistant message with new content
 */
function updateAssistantMessage(msg, content) {
    const contentDiv = msg.querySelector('.content');
    contentDiv.textContent = content;
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

/**
 * Display sources in the sidebar
 */
function displaySources(sources) {
    currentSources = sources;
    elements.sourcesList.innerHTML = '';

    if (!sources || sources.length === 0) {
        elements.sourcesList.innerHTML = '<p class="no-sources">No sources found.</p>';
        return;
    }

    sources.forEach((source, index) => {
        const card = document.createElement('div');
        card.className = 'source-card';
        card.dataset.index = index;

        const title = document.createElement('div');
        title.className = 'source-title';
        title.textContent = source.page_title || `Source ${index + 1}`;

        const snippet = document.createElement('div');
        snippet.className = 'source-snippet';
        snippet.textContent = source.text ? source.text.substring(0, 200) + '...' : '';

        const score = document.createElement('div');
        score.className = 'source-score';
        score.textContent = source.score ? `Score: ${source.score.toFixed(3)}` : '';

        card.appendChild(title);
        card.appendChild(snippet);
        if (source.score) card.appendChild(score);

        card.addEventListener('click', () => highlightSource(index));

        elements.sourcesList.appendChild(card);
    });
}

/**
 * Highlight a source card
 */
function highlightSource(index) {
    const cards = elements.sourcesList.querySelectorAll('.source-card');
    cards.forEach((card, i) => {
        if (i === index) {
            card.classList.add('highlighted');
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            card.classList.remove('highlighted');
        }
    });
}

/**
 * Send a chat message via SSE streaming
 */
async function sendMessage(query) {
    if (isStreaming) return;
    isStreaming = true;

    // Add user message
    addMessage('user', query);

    // Add loading assistant message
    const assistantMsg = addMessage('assistant', '', true);

    // Clear sources
    elements.sourcesList.innerHTML = '<p class="no-sources">Searching...</p>';

    // Disable input
    elements.sendBtn.disabled = true;
    elements.input.value = '';

    let fullResponse = '';

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const contentType = response.headers.get('content-type');

        if (contentType && contentType.includes('text/event-stream')) {
            // Handle SSE streaming
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;

                        try {
                            const parsed = JSON.parse(data);

                            if (parsed.type === 'sources') {
                                displaySources(parsed.sources);
                            } else if (parsed.type === 'token') {
                                fullResponse += parsed.token;
                                assistantMsg.classList.remove('loading');
                                updateAssistantMessage(assistantMsg, fullResponse);
                            } else if (parsed.type === 'error') {
                                throw new Error(parsed.message);
                            }
                        } catch (e) {
                            if (e instanceof SyntaxError) {
                                // Not valid JSON, might be partial
                                console.warn('Invalid JSON:', data);
                            } else {
                                throw e;
                            }
                        }
                    }
                }
            }
        } else {
            // Handle regular JSON response (fallback)
            const data = await response.json();

            if (data.sources) {
                displaySources(data.sources);
            }

            if (data.response) {
                fullResponse = data.response;
                assistantMsg.classList.remove('loading');
                updateAssistantMessage(assistantMsg, fullResponse);
            }
        }

    } catch (error) {
        console.error('Chat error:', error);
        assistantMsg.classList.remove('loading');
        updateAssistantMessage(assistantMsg, `Error: ${error.message}`);
        elements.sourcesList.innerHTML = '<p class="no-sources">Error loading sources.</p>';
    } finally {
        isStreaming = false;
        elements.sendBtn.disabled = false;
        elements.input.focus();
    }
}

/**
 * Handle form submission
 */
elements.form.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = elements.input.value.trim();
    if (query) {
        sendMessage(query);
    }
});

/**
 * Handle keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Focus input on any key press when not focused
    if (!['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
            elements.input.focus();
        }
    }
});
