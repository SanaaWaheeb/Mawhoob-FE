import { Component } from '@angular/core';
import {Router} from "@angular/router";

@Component({
  selector: 'app-sign-in',
  standalone: false,
  templateUrl: './sign-in.component.html',
  styleUrl: './sign-in.component.scss'
})
export class SignInComponent {
  constructor(private router: Router) {
  }

  onNavigateMain() {
    this.router.navigate(['main', 'home']);
  }
}
