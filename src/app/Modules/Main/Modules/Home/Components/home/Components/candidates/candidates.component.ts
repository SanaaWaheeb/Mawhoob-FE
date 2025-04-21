import {Component} from '@angular/core';

@Component({
    selector: 'app-candidates',
    standalone: false,
    templateUrl: './candidates.component.html',
    styleUrl: './candidates.component.scss'
})
export class CandidatesComponent {
    candidatesList = [
        {name: 'Omar Khalid Al-Fahad', value: 90},
        {name: 'Yousef Nasser Al-Dossari', value: 87},
        {name: 'Fahad Saleh Al-Qahtani', value: 82},
        {name: 'Abdullah Majed Al-Shammari', value: 76},
        {name: 'Saif Ahmed Al-Mutairi', value: 75},
        {name: 'Rayan Sami Al-Harbi', value: 75},
        {name: 'Nayef Tariq Al-Subaie', value: 65},
    ];
}
