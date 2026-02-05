export function formatRelativeTime(isoString) {
    if (!isoString) return '';
    
    // Server returns UTC times without 'Z' suffix - add it if missing
    let utcString = isoString;
    if (!isoString.endsWith('Z') && !isoString.includes('+') && !isoString.includes('-', 10)) {
        utcString = isoString + 'Z';
    }
    const date = new Date(utcString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);

    if (diffSec < 0) return 'just now';
    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffSec < 3600) {
        const mins = Math.floor(diffSec / 60);
        const secs = diffSec % 60;
        return secs > 0 ? `${mins}m ${secs}s ago` : `${mins}m ago`;
    }
    if (diffSec < 86400) {
        const hours = Math.floor(diffSec / 3600);
        const mins = Math.floor((diffSec % 3600) / 60);
        return mins > 0 ? `${hours}h ${mins}m ago` : `${hours}h ago`;
    }
    return `${Math.floor(diffSec / 86400)}d ago`;
}
