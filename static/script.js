/* ======================================================
   GET USER RECOMMENDATIONS
====================================================== */

async function getRecommendations() {
  const userId = document.getElementById("userId").value;
  if (!userId) return alert("Please enter a User ID");

  const spinner = document.getElementById("loadingSpinner");
  spinner.classList.remove("hidden");

  try {
    const response = await fetch(`/recommend/user?user_id=${userId}&n=8`);
    const data = await response.json();

    if (data.error) {
      alert("❌ " + data.error);
      return;
    }

    renderBooks(data["SVD++_Recommendations"], "svdppCards");
    renderBooks(data["BPR_Recommendations"], "bprCards");


  } catch (err) {
    alert("❌ Failed to load recommendations:\n" + err);
  } finally {
    spinner.classList.add("hidden");
  }
}

/* ======================================================
   RENDER BOOK CARDS
====================================================== */

function renderBooks(books, containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = books.map(book => `
    <div class="book-card bg-gray-800/70 rounded-xl shadow-lg p-4 cursor-pointer 
                transform transition hover:shadow-xl hover:scale-105 duration-300">

      <img 
        src="${book.image_url || '/static/assets/no_cover.png'}"
        alt="${book.title}" 
        referrerpolicy="no-referrer"
        onerror="this.onerror=null; this.src='/static/assets/no_cover.png';"
        class="w-full h-80 object-cover rounded-2xl shadow-md mb-3 
               transition-transform duration-300 hover:scale-105"
      />

      <h3 class="text-lg font-semibold text-indigo-300 truncate">${book.title}</h3>
      <p class="text-gray-400 truncate">by ${book.author}</p>
      <p class="text-yellow-400 mt-1">⭐ ${parseFloat(book.rating).toFixed(2)}</p>

      <button 
        onclick='viewMore(${JSON.stringify(book).replace(/"/g, "&quot;")})'
        class="mt-3 bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded-lg transition-colors">
        View More
      </button>
    </div>
  `).join("");

  // Fade-in animation
  document.querySelectorAll(".book-card img").forEach(img => {
    img.addEventListener("load", () => img.classList.add("loaded"));
  });
}

/* ======================================================
   VIEW MORE (MODAL POPUP)
====================================================== */

function viewMore(book) {
  document.getElementById("modalImage").src = book.image_url || '/static/assets/no_cover.png';
  document.getElementById("modalTitle").innerText = book.title;
  document.getElementById("modalAuthor").innerText = "by " + book.author;
  document.getElementById("modalRating").innerText = "⭐ " + book.rating;
  document.getElementById("modalDesc").innerText = book.description || "No description available.";

  document.getElementById("bookModal").classList.remove("hidden");

  const btn = document.getElementById("similarBtn");

  btn.onclick = async () => {
    btn.innerText = "Loading similar books...";
    try {
      const r = await fetch(`/recommend/similar?book_id=${book.id}&n=5`);
      const d = await r.json();

      if (d.error) {
        document.getElementById("modalDesc").innerHTML = "❌ " + d.error;
      } else {
        const htmlList = d.Similar_Books.map(b =>
          `<li class="mt-1">📘 <b>${b.title}</b> <i>by ${b.author}</i></li>`
        ).join("");

        document.getElementById("modalDesc").innerHTML =
          `<b>Similar Books:</b><ul class="mt-2">${htmlList}</ul>`;
      }

    } catch (err) {
      document.getElementById("modalDesc").innerHTML =
        "❌ Error fetching similar books.";
    }

    btn.innerText = "Find Similar Books";
  };
}

/* ======================================================
   CLOSE MODAL
====================================================== */

function closeModal() {
  document.getElementById("bookModal").classList.add("hidden");
}

/* ======================================================
   PAGINATION (Show More)
====================================================== */

let visibleCount = { svdppCards: 8, bprCards: 8 };

function showMore(section) {
  visibleCount[section] += 8;
  const cards = document.querySelectorAll(`#${section} .book-card`);

  cards.forEach((card, index) => {
    card.style.display = index < visibleCount[section] ? "block" : "none";
  });
}
