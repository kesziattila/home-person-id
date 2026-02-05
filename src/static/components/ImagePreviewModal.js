export default {
    template: `
        <div v-if="visible" @click.self="close" class="modal active fixed inset-0 bg-black bg-opacity-90 items-center justify-center z-50 p-4">
            <div class="relative max-w-4xl w-full">
                <button @click="close" class="absolute top-0 right-0 text-white text-3xl p-4">&times;</button>
                <img :src="imageUrl" class="max-w-full max-h-[80vh] mx-auto rounded-lg">
                <div v-if="showDelete" class="text-center mt-4">
                    <button @click="deleteImage" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-sm">
                        Delete Image
                    </button>
                </div>
            </div>
        </div>
    `,
    props: {
        imageUrl: {
            type: String,
            default: null,
        },
        showDelete: {
            type: Boolean,
            default: false,
        }
    },
    emits: ['close', 'delete-image'],
    data() {
        return {
            visible: false,
        };
    },
    watch: {
        imageUrl(newUrl) {
            this.visible = !!newUrl;
        }
    },
    methods: {
        close() {
            this.$emit('close');
        },
        deleteImage() {
            this.$emit('delete-image');
        }
    }
};
