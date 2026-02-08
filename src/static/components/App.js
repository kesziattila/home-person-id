import Streams from './Streams.js';
import Persons from './Persons.js';
import Events from './Events.js';
import Tracks from './Tracks.js';
import Unidentified from './Unidentified.js';
import PersonDetailModal from './PersonDetailModal.js';
import AssignFaceModal from './AssignFaceModal.js';
import EventDetailModal from './EventDetailModal.js';
import AddPersonModal from './AddPersonModal.js';
import ImagePreviewModal from './ImagePreviewModal.js';

export default {
    components: {
        Streams,
        Persons,
        Events,
        Tracks,
        Unidentified,
        PersonDetailModal,
        AssignFaceModal,
        EventDetailModal,
        AddPersonModal,
        ImagePreviewModal,
    },
    template: `
        <div>
            <nav class="bg-gray-800 p-4 shadow-lg">
                <div class="container mx-auto flex justify-between items-center">
                    <h1 class="text-xl font-bold">Home Person ID</h1>
                    <div id="status-badge" class="px-3 py-1 rounded-full text-xs font-medium bg-green-500 text-green-950">
                        System Online
                    </div>
                    <div class="flex items-center gap-2 ml-4">
                        <input type="checkbox" id="auto-refresh-check" v-model="autoRefresh" class="rounded border-gray-600 bg-gray-700">
                        <label for="auto-refresh-check" class="text-xs text-gray-400 cursor-pointer select-none">Auto Refresh</label>
                    </div>
                </div>
            </nav>

            <div class="bg-gray-800 border-b border-gray-700">
                <div class="container mx-auto flex space-x-2">
                    <button v-for="tab in tabs" :key="tab.id" 
                            @click="activeTab = tab.id"
                            :class="['tab-button', { 'active': activeTab === tab.id }]">
                        {{ tab.name }}
                    </button>
                </div>
            </div>

            <main class="container mx-auto p-6">
                <component :is="activeTabComponent"
                           :auto-refresh="autoRefresh"
                           :apply-filter="eventFilter"
                           @show-person-detail="showPersonDetail"
                           @show-assign-face="showAssignFace"
                           @show-event-detail="showEventDetail"
                           @show-add-person="showAddPersonModal = true"
                           @preview-image="showImagePreview"
                           @face-assigned="handleFaceAssigned"
                           @person-deleted="handlePersonDeleted"
                           @filter-applied="eventFilter = null">
                </component>
            </main>

            <!-- Modals -->
            <add-person-modal :visible="showAddPersonModal" @close="showAddPersonModal = false" @person-added="handlePersonAdded"></add-person-modal>
            
            <person-detail-modal :person-id="selectedPersonId" 
                                 @close="selectedPersonId = null"
                                 @person-deleted="handlePersonDeleted"
                                 @preview-image="showImagePreview">
            </person-detail-modal>

            <assign-face-modal :face-id="selectedFaceId"
                               @close="selectedFaceId = null"
                               @face-assigned="handleFaceAssigned">
            </assign-face-modal>

            <event-detail-modal :event="selectedEvent"
                                @close="selectedEvent = null"
                                @filter-events="applyEventFilter">
            </event-detail-modal>

            <image-preview-modal :image-url="previewImageUrl"
                                 @close="previewImageUrl = null">
            </image-preview-modal>

        </div>
    `,
    data() {
        return {
            activeTab: 'streams',
            autoRefresh: localStorage.getItem('autoRefresh') !== 'false',
            tabs: [
                { id: 'streams', name: 'Live Streams' },
                { id: 'persons', name: 'Persons' },
                { id: 'unidentified', name: 'Unidentified' },
                { id: 'events', name: 'Events' },
                { id: 'tracks', name: 'Active Tracks' },
            ],
            showAddPersonModal: false,
            selectedPersonId: null,
            selectedFaceId: null,
            selectedEvent: null,
            eventFilter: null,
            previewImageUrl: null,
        };
    },
    computed: {
        activeTabComponent() {
            const tabExists = this.tabs.some(tab => tab.id === this.activeTab);
            if (!tabExists) return null;
            return this.activeTab.charAt(0).toUpperCase() + this.activeTab.slice(1);
        }
    },
    watch: {
        activeTab(newTab) {
            if (window.location.hash !== '#' + newTab) {
                window.location.hash = newTab;
            }
        },
        autoRefresh(newVal) {
            localStorage.setItem('autoRefresh', newVal);
        }
    },
    created() {
        const hash = window.location.hash.replace('#', '');
        if (this.tabs.some(tab => tab.id === hash)) {
            this.activeTab = hash;
        }

        window.addEventListener('hashchange', () => {
            const newHash = window.location.hash.replace('#', '');
            if (this.tabs.some(tab => tab.id === newHash)) {
                this.activeTab = newHash;
            }
        });

        // Global Escape key listener
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
    },
    methods: {
        closeAllModals() {
            this.showAddPersonModal = false;
            this.selectedPersonId = null;
            this.selectedFaceId = null;
            this.selectedEvent = null;
            this.previewImageUrl = null;
        },
        showPersonDetail(personId) {
            this.selectedPersonId = personId;
        },
        showAssignFace(faceId) {
            this.selectedFaceId = faceId;
        },
        showEventDetail(event) {
            this.selectedEvent = event;
        },
        showImagePreview(imageUrl) {
            this.previewImageUrl = imageUrl;
        },
        handlePersonAdded() {
            if (this.activeTab === 'persons') {
                this.forceRerender(this.activeTab);
            }
        },
        handlePersonDeleted() {
            if (this.activeTab === 'persons') {
                this.forceRerender(this.activeTab);
            }
        },
        applyEventFilter({ field, value }) {
            this.eventFilter = { field, value };
            this.selectedEvent = null;
            this.activeTab = 'events';
        },
        handleFaceAssigned() {
            if (this.activeTab === 'unidentified') {
                this.forceRerender(this.activeTab);
            }
        },
        forceRerender(tabName) {
            this.activeTab = '';
            this.$nextTick(() => {
                this.activeTab = tabName;
            });
        }
    }
};
