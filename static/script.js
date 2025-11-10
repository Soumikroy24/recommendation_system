async function getRecommendations() {
  const userId = document.getElementById('userId').value;
  if (!userId) return alert("Please enter a User ID");

  document.getElementById('loadingSpinner').classList.remove('hidden');
  try {
    const response = await fetch(`/recommend/user?user_id=${userId}&n=8`);
    const data = await response.json();
    renderBooks(data["SVD++_Recommendations"], "svdppCards");
    renderBooks(data["BPR_Recommendations"], "bprCards");
  } finally {
    document.getElementById('loadingSpinner').classList.add('hidden');
  }
}

function renderBooks(books, containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = books.map(book => `
    <div class="book-card bg-gray-800 rounded-xl shadow-lg p-4 cursor-pointer transform transition hover:shadow-xl hover:scale-105 duration-300">
      <img 
  src="${book.image_url || '/static/assets/no_cover.png'}"
  alt="${book.title}" 
  referrerpolicy="no-referrer"
  onerror="this.onerror=null; this.src='/static/assets/no_cover.png'; this.nextElementSibling.innerText='⚠️ Image not available';"
  class="w-full h-80 object-cover rounded-2xl shadow-md mb-3 transition-transform duration-300 hover:scale-105"
/>

      <h3 class="text-lg font-semibold text-indigo-300 truncate">${book.title}</h3>
      <p class="text-gray-400 truncate">by ${book.author}</p>
      <p class="text-yellow-400 mt-1">⭐ ${book.rating.toFixed(2)}</p>
      <button onclick='viewMore(${JSON.stringify(book).replace(/"/g, "&quot;")})'
              class="mt-3 bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded-lg transition-colors">
        View More
      </button>
    </div>
  `).join('');

  //  Smooth fade-in effect for all images
  document.querySelectorAll('.book-card img').forEach(img => {
    img.addEventListener('load', () => {
      img.classList.add('loaded');
    });
  });
}


function viewMore(book) {
  // Fill modal with book info
  document.getElementById('modalImage').src = book.image_url || 'https://via.placeholder.com/200x300';
  document.getElementById('modalTitle').innerText = book.title;
  document.getElementById('modalAuthor').innerText = "by " + book.author;
  document.getElementById('modalRating').innerText = "⭐ " + book.rating;
  document.getElementById('modalDesc').innerText = book.description || "No description available.";
  document.getElementById('bookModal').classList.remove('hidden');

  // Attach similar-book handler
  const btn = document.getElementById('similarBtn');
  btn.onclick = async () => {
    btn.innerText = "Loading similar books...";
    const res = await fetch(`/recommend/similar?book_id=${book.id}&n=5`);
    const data = await res.json();

    if (data.Similar_Books) {
      document.getElementById('modalDesc').innerHTML =
        `<b>Similar Books:</b><br>${data.Similar_Books.join('<br>')}`;
    } else {
      document.getElementById('modalDesc').innerHTML =
        "❌ No similar books found.";
    }
    btn.innerText = "Find Similar Books";
  };
}


function closeModal() {
  document.getElementById('bookModal').classList.add('hidden');
}

// ===== PAGINATION LOGIC =====
let visibleCount = { svdppCards: 8, bprCards: 8 };

function showMore(section) {
  visibleCount[section] += 8;  // Show 8 more each time
  const cards = document.querySelectorAll(`#${section} .book-card`);
  cards.forEach((card, i) => {
    card.style.display = i < visibleCount[section] ? "block" : "none";
  });
}
